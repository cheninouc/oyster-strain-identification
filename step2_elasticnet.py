#!/usr/bin/env python3
"""step2: one-vs-rest Elastic Net 稳定筛选（对应 Supplementary Methods S1.2）。

对每个选育系：
  1) 取 step1 为该品系保留的候选 SNP（默认 top 2,000）；
  2) 以该品系为阳性类、其余 6 个选育系 + Wild 为阴性类，拟合 one-vs-rest Elastic Net
     逻辑回归（SAGA、balanced class weights、additive dosage）；
  3) 在全体 discovery 个体上用分层 5 折 CV 确定 (C, l1_ratio)，准则为 mean MCC
     加 one-standard-error sparse-model 规则；该超参数随后固定；
  4) 用固定超参数做 B 次分层随机子采样（默认 100 次 × 80% 个体），统计每个 SNP
     的非零系数选择频率；
  5) 按选择频率降序（并列用平均 |coef| 降序）排序，取每个品系 top K（默认 200）；
  6) 合并各品系 top K 并去重，得到候选并集（理想情况 7 × 200 = 1,400）。

预处理（中位数填补 + 标准化）全部放在 sklearn Pipeline 内，参数只用训练折估计，
留出数据不参与拟合，避免信息泄漏。

用法：
  python3 step2_elasticnet.py --indir ../step1/out --raw geno.raw --pop ../vcf/sample_pop \
      --strains CF JF KF SB SG SO SW --top-k 200 --outdir out
"""
import argparse
import math
import os
import sys
import time
import warnings

import numpy as np
import re

import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
L1_GRID = [0.25, 0.50, 0.75, 0.90, 1.00]
NZ_TOL = 1e-8            # 系数绝对值超过该阈值视为“被选中”（与 S1.2 一致）
STRAIN_LABEL = {'CF': 'CS', 'JF': 'JS', 'KF': 'KS', 'SB': 'BS',
                'SG': 'GS', 'SO': 'OS', 'SW': 'WS'}

# scikit-learn 1.8 起 penalty='elasticnet' 被弃用，改为只在 l1_ratio 上设 0<r<1；
# 旧版本仍需显式 penalty='elasticnet'。这里按版本分支，避免警告刷屏。
_SK_VER = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


# ---------------------------------------------------------------- 输入
def read_raw(path):
    """读 plink --recode A 的 .raw 文件 -> (X DataFrame, 含 FID/IID 的原始表)。"""
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    for c in ('FID', 'IID'):
        if c not in df.columns:
            raise SystemExit(f'{path}: 缺少 {c} 列')
    geno = df.drop(columns=[c for c in ('FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE')
                            if c in df.columns])
    geno = geno.drop(columns=['IID'])
    # plink 列名为 <SNP ID>_<A1>，SNP ID 形如 NC_047559.1:2331。
    # 仅在结尾确为单碱基时才剥离，避免把 ID 自带的 '_' 也切掉。
    geno.columns = [c[:-2] if re.search(r'_[ACGT]$', c) else c
                    for c in geno.columns]
    X = geno.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    return X, df[['FID', 'IID']].astype(str)


def align_pop(raw_ids, pop):
    """把 sample_pop 的群体标签对齐到 .raw 的样本行。

    plink --vcf 导入时默认按第一个 '_' 把样本名拆成 FID+IID，因此 .raw 里的
    IID 可能已被截断（CF14_8 -> FID=CF14, IID=8）。这里按 IID、FID_IID、
    FID-IID、'/'互换 的顺序回退匹配，并统计各路径命中数。
    """
    known = dict(zip(pop['IID'].astype(str), pop['POP'].astype(str)))
    out, used = [], []
    via = {'IID': 0, 'FID_IID': 0, 'hyphen': 0, 'unmatched': 0}
    for fid, iid in zip(raw_ids['FID'], raw_ids['IID']):
        if iid in known:
            out.append(known[iid]); used.append(iid); via['IID'] += 1; continue
        hit, path, key = None, None, None
        if fid not in ('0', '', 'nan', iid):
            for k in (f'{fid}_{iid}', f'{fid}-{iid}'):
                if k in known:
                    hit, path, key = known[k], 'FID_IID', k; break
        if hit is None:
            for k in (iid.replace('-', '_'), iid.replace('_', '-')):
                if k in known:
                    hit, path, key = known[k], 'hyphen', k; break
        if hit is None:
            out.append(np.nan); used.append(None); via['unmatched'] += 1
        else:
            out.append(hit); used.append(key); via[path] += 1
    return pd.Series(out), via, used


def find_top_file(outdir, strain, top_k):
    """定位 {strain}_top*.tsv。实际保留数 k = min(top_k, 候选数)，
    因此文件名中的 k 未必等于 top_k，按 k 精确匹配优先、否则取最大的一个。
    （注意不能直接按字典序取首个：CF_top200.tsv < CF_top2000.tsv。）"""
    hits = [f for f in os.listdir(outdir)
            if f.startswith(strain + '_top') and f.endswith('.tsv')]
    if not hits:
        return None
    want = f'{strain}_top{top_k}.tsv'
    if want in hits:
        return os.path.join(outdir, want)

    def k_of(f):
        try:
            return int(f[len(strain) + 4:-4])
        except ValueError:
            return -1
    return os.path.join(outdir, max(hits, key=k_of))


def read_candidates(indir, strain, top_n):
    """读 step1 的 {strain}_top*.tsv，返回按排名排序的 marker 列表。"""
    path = find_top_file(indir, strain, top_n)
    if path is None:
        raise SystemExit(f'{indir}: 找不到 {strain}_top*.tsv')
    d = pd.read_csv(path, sep='\t')
    if 'marker' not in d.columns:
        raise SystemExit(f'{path}: 缺少 marker 列')
    if 'rank' in d.columns:
        d = d.sort_values('rank')
    markers = d['marker'].astype(str).tolist()
    if top_n and top_n < len(markers):
        markers = markers[:top_n]
    return markers


def read_pop(path):
    pop = pd.read_csv(path, sep=r'\s+', header=None, names=['IID', 'POP'], dtype=str)
    return pop


def make_labels(grp, strain):
    """返回 y（1 = target strain，0 = 其余 6 品系 + Wild）。"""
    if grp.isna().any():
        miss = grp[grp.isna()].index.tolist()[:5]
        raise SystemExit(f'仍有 {int(grp.isna().sum())} 个样本无法匹配，例如 {miss}')
    y = (grp.values == strain).astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 5:
        raise SystemExit(f'{strain}: 阳性样本仅 {n_pos} 个，无法做 5 折 CV')
    return y, n_pos, n_neg


# ---------------------------------------------------------------- 建模
def build_pipeline(C, l1, seed):
    """中位数填补 + 标准化 + Elastic Net 逻辑回归（全部吸入 Pipeline）。"""
    kw = dict(solver='saga', C=C, l1_ratio=l1, class_weight='balanced',
              max_iter=10000, tol=1e-4, random_state=seed)
    if _SK_VER < (1, 8):
        kw['penalty'] = 'elasticnet'
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(**kw)),
    ])


def fold_metrics(y_true, y_pred, y_prob):
    """一个测试折上的 9 个指标（与 S1.2 列出的指标一致）。"""
    out = {
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    try:
        out['roc_auc'] = roc_auc_score(y_true, y_prob)
        out['pr_auc'] = average_precision_score(y_true, y_prob)
    except ValueError:            # 测试折只有单一类别
        out['roc_auc'] = np.nan
        out['pr_auc'] = np.nan
    return out


def eval_combo(X, y, folds, C, l1, seed):
    """某个 (C, l1_ratio) 在全部 CV 折上的指标与非零系数数。"""
    rows = []
    for tr, te in folds:
        pipe = build_pipeline(C, l1, seed)
        pipe.fit(X[tr], y[tr])
        clf = pipe.named_steps['clf']
        prob = pipe.predict_proba(X[te])[:, 1]
        m = fold_metrics(y[te], (prob >= 0.5).astype(int), prob)
        m['nnz'] = int(np.sum(np.abs(clf.coef_[0]) > NZ_TOL))
        rows.append(m)
    d = pd.DataFrame(rows)
    res = {f'mean_{k}': float(d[k].mean()) for k in d.columns}
    res['se_mcc'] = float(d['mcc'].std(ddof=1) / math.sqrt(len(d)))
    res['C'] = C
    res['l1_ratio'] = l1
    return res


def select_hyperparams(tab):
    """S1.2 的 one-standard-error sparse-model 规则。"""
    tab = tab.copy()
    best = tab.loc[tab['mean_mcc'].idxmax()]
    # MCC 并列时用 balanced accuracy 排序
    tie = tab[np.isclose(tab['mean_mcc'], best['mean_mcc'], atol=1e-12)]
    if len(tie) > 1:
        best = tie.loc[tie['mean_balanced_accuracy'].idxmax()]
    thr = best['mean_mcc'] - best['se_mcc']

    comp = tab[tab['mean_mcc'] >= thr]
    if (comp['mean_nnz'] > 0).any():
        comp = comp[comp['mean_nnz'] > 0]
    # 最稀疏 -> 较大 l1_ratio -> 较小 C -> 较高 MCC
    comp = comp.assign(_k=list(zip(comp['mean_nnz'], -comp['l1_ratio'],
                                   comp['C'], -comp['mean_mcc'])))
    pick = comp.sort_values('_k').iloc[0]
    return pick, thr


def _subsample_fit(X, y, tr, C, l1, seed):
    """一次子采样拟合，返回 (是否非零 0/1 向量, |coef| 向量)。"""
    pipe = build_pipeline(C, l1, seed)
    pipe.fit(X[tr], y[tr])
    coef = pipe.named_steps['clf'].coef_[0]
    return (np.abs(coef) > NZ_TOL).astype(float), np.abs(coef)


def nonzero_freq(X, y, C, l1, n_sub, frac, seed, n_jobs=1):
    """B 次分层子采样下的选择频率与平均 |coef|（子采样之间并行）。"""
    p = X.shape[1]
    freq = np.zeros(p)
    absum = np.zeros(p)
    sss = StratifiedShuffleSplit(n_splits=n_sub, test_size=1.0 - frac,
                                  random_state=seed)
    # 先一次性生成全部划分：StratifiedShuffleSplit 由 random_state 决定，
    # 与并行顺序无关，因此并行与串行结果完全一致
    splits = [tr for tr, _ in sss.split(X, y)]
    got = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_subsample_fit)(X, y, tr, C, l1, seed) for tr in splits)
    for nz, ac in got:
        freq += nz
        absum += ac
    used = len(got)
    return freq / used, absum / used, used


# ---------------------------------------------------------------- 主流程
def run_strain(strain, markers, Xall, grp, args):
    y, n_pos, n_neg = make_labels(grp, strain)

    have = [m for m in markers if m in Xall.columns]
    if len(have) < len(markers):
        print(f'  警告: {strain} 有 {len(markers) - len(have)} 个候选 SNP 不在基因型文件中')
    X = Xall[have].values
    print(f'  {strain}: 阳性 {n_pos} / 阴性 {n_neg}，候选 SNP {len(have)}，'
          f'缺失率 {np.isnan(X).mean():.4f}')

    folds = list(StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                                 random_state=args.seed).split(X, y))

    combos = [(c, l) for c in C_GRID for l in L1_GRID]
    res = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(eval_combo)(X, y, folds, c, l, args.seed) for c, l in combos)
    tab = pd.DataFrame(res)

    pick, thr = select_hyperparams(tab)
    C, l1 = float(pick['C']), float(pick['l1_ratio'])
    print(f'  选定超参数: C={C:g}, l1_ratio={l1:g} '
          f'(mean MCC={pick["mean_mcc"]:.3f}, 1SE 阈值={thr:.3f}, '
          f'mean nnz={int(pick["mean_nnz"])})')

    freq, absmean, used = nonzero_freq(X, y, C, l1, args.n_subsample,
                                       args.subsample_frac, args.seed,
                                       n_jobs=args.n_jobs)

    d = pd.DataFrame({'strain': strain, 'marker': have,
                      'selection_frequency': freq, 'mean_abs_coef': absmean})
    d['selected'] = d['selection_frequency'] > 0
    d = d.sort_values(['selection_frequency', 'mean_abs_coef', 'marker'],
                      ascending=[False, False, True]).reset_index(drop=True)
    d['rank'] = np.arange(1, len(d) + 1)

    k = min(args.top_k, len(d))
    d.to_csv(os.path.join(args.outdir, f'{strain}_stability.tsv'),
             sep='\t', index=False, float_format='%.6g')
    top = d.head(k)
    top.to_csv(os.path.join(args.outdir, f'{strain}_top{k}.tsv'),
               sep='\t', index=False, float_format='%.6g')

    if pick['mean_nnz'] == 0:
        print(f'  警告: {strain} 选中的超参数在所有 CV 折上均无保留变量，'
              f'该品系的 top {k} 仅为并列排名，选择频率无区分度')
    elif top['selection_frequency'].min() == 0:
        print(f'  警告: {strain} 前 {k} 名中已有选择频率为 0 的 SNP，'
              f'名次之间存在并列')

    hp = pd.DataFrame([{'strain': strain, 'C': C, 'l1_ratio': l1,
                        'mean_mcc': pick['mean_mcc'], 'se_mcc': pick['se_mcc'],
                        'one_se_threshold': thr, 'mean_nnz': int(pick['mean_nnz']),
                        'n_positive': n_pos, 'n_negative': n_neg,
                        'n_candidate': len(have), 'n_subsample': used,
                        'subsample_frac': args.subsample_frac,
                        'n_ever_selected': int(d['selected'].sum())}])
    # 每品系单独存一份超参数，供 --merge-only 汇总（并行运行时各品系独立写文件）
    hp.to_csv(os.path.join(args.outdir, f'hp_{strain}.tsv'), sep='\t',
              index=False, float_format='%.6g')
    print(f'  {used} 次子采样中被选过至少一次的 SNP: {int(d["selected"].sum())}；'
          f'前 {k} 名的最低选择频率 = {top["selection_frequency"].min():.3f}')
    return top[['strain', 'marker', 'selection_frequency', 'mean_abs_coef',
                'rank']], hp


def find_top_file(outdir, strain, top_k):
    """定位 {strain}_top*.tsv。实际保留数 k = min(top_k, 候选数)，
    因此文件名中的 k 未必等于 top_k，按 k 精确匹配优先、否则取最大的一个。"""
    hits = [f for f in os.listdir(outdir)
            if f.startswith(strain + '_top') and f.endswith('.tsv')]
    if not hits:
        return None
    want = f'{strain}_top{top_k}.tsv'
    if want in hits:
        return os.path.join(outdir, want)
    def k_of(f):
        try:
            return int(f[len(strain) + 4:-4])
        except ValueError:
            return -1
    return os.path.join(outdir, max(hits, key=k_of))


def merge_outputs(strains, outdir, top_k):
    """合并各品系的 top-k：写 strain_selected_snps.tsv 与去重后的 candidates_union.tsv。"""
    sel, hps, missing = [], [], []
    for s in strains:
        f = find_top_file(outdir, s, top_k)
        if f is None:
            missing.append(s)
            continue
        d = pd.read_csv(f, sep='\t')
        d['strain_label'] = d['strain'].map(lambda x: STRAIN_LABEL.get(x, x))
        sel.append(d)
        h = os.path.join(outdir, f'hp_{s}.tsv')
        if os.path.exists(h):
            hps.append(pd.read_csv(h, sep='\t'))
    if missing:
        raise SystemExit('以下品系缺少结果文件，无法合并: ' + ', '.join(missing))

    sel = pd.concat(sel, ignore_index=True)
    sel.to_csv(os.path.join(outdir, 'strain_selected_snps.tsv'),
               sep='\t', index=False, float_format='%.6g')
    if hps:
        pd.concat(hps, ignore_index=True).to_csv(
            os.path.join(outdir, 'hyperparameters.tsv'), sep='\t',
            index=False, float_format='%.6g')

    g = (sel.groupby('marker')
         .agg(n_strains=('strain', 'nunique'),
              strains=('strain_label', lambda v: ','.join(sorted(set(v)))),
              best_rank=('rank', 'min'),
              max_freq=('selection_frequency', 'max'))
         .reset_index()
         .sort_values(['n_strains', 'best_rank'], ascending=[False, True]))
    g.to_csv(os.path.join(outdir, 'candidates_union.tsv'),
             sep='\t', index=False, float_format='%.6g')

    print('\n' + '=' * 62)
    print(f'各品系保留 {top_k} 个: 合计 {len(sel)} 条记录')
    print(f'去重后唯一 SNP: {len(g)}（理论上限 {len(strains) * top_k}）')
    multi = g[g['n_strains'] > 1]
    print(f'被 ≥2 个品系同时选中: {len(multi)} 个 SNP')
    if len(multi):
        cnt = multi['n_strains'].value_counts().sort_index()
        print('  按共享品系数:' + ', '.join(f'{k}品系={v}' for k, v in cnt.items()))
    print(f'结果目录: {outdir}/')


def main():
    p = argparse.ArgumentParser(
        description='one-vs-rest Elastic Net 稳定筛选（S1.2）')
    p.add_argument('--indir', required=True, help='step1 输出目录（含 {strain}_top*.tsv）')
    p.add_argument('--raw', help='plink --recode A 生成的 .raw 基因型文件')
    p.add_argument('--pop', help='sample_pop（IID\\tPOP）')
    p.add_argument('--strains', nargs='+', default=['CF', 'JF', 'KF', 'SB', 'SG', 'SO', 'SW'])
    p.add_argument('--only-strain', metavar='S',
                   help='只处理这一个品系（用于 7 品系并行；不写合并结果）')
    p.add_argument('--merge-only', action='store_true',
                   help='跳过建模，仅合并已有的各品系结果')
    p.add_argument('--cand-top-n', type=int, default=2000,
                   help='取 step1 每品系前多少个 SNP 作为候选（默认 2000）')
    p.add_argument('--top-k', type=int, default=200, help='每品系最终保留数（默认 200）')
    p.add_argument('--n-subsample', type=int, default=100, help='子采样次数 B')
    p.add_argument('--subsample-frac', type=float, default=0.80, help='每次子采样比例')
    p.add_argument('--cv-folds', type=int, default=5, help='超参数 CV 折数')
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='本进程内并行数（-1 为全部核心；7 品系并行时请设为核数/7）')
    p.add_argument('--outdir', default='out')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 仅合并 ----
    if args.merge_only:
        merge_outputs(args.strains, args.outdir, args.top_k)
        return

    if not args.raw or not args.pop:
        raise SystemExit('需要 --raw 与 --pop（或在 --merge-only 模式下省略）')

    Xall, raw_ids = read_raw(args.raw)
    pop = read_pop(args.pop)
    grp, via, matched = align_pop(raw_ids, pop)
    print(f'基因型矩阵: {Xall.shape[0]} 样本 × {Xall.shape[1]} SNP')
    print(f'样本对齐: IID 直接匹配 {via["IID"]}，FID 拼接匹配 {via["FID_IID"]}，'
          f'分隔符回退 {via["hyphen"]}，未匹配 {via["unmatched"]}')

    # 基因型文件里多出、但不在 sample_pop 中的样本：剔除
    if via['unmatched']:
        bad = raw_ids[grp.isna()].head(5)
        print('  已剔除（不在 sample_pop 中）:',
              '; '.join(f'{a}|{b}' for a, b in zip(bad['FID'], bad['IID'])),
              '…' if via['unmatched'] > 5 else '')
        keep = grp.notna().values
        Xall = Xall[keep].reset_index(drop=True)
        grp = grp[keep].reset_index(drop=True)

    # 反向核对：sample_pop 中应有的人是否都在基因型文件里（仅提示，不中断）
    covered = {k for k in matched if k}
    absent = sorted(set(pop['IID'].astype(str)) - covered)
    if absent:
        print(f'  提示: sample_pop 的 {len(pop)} 个样本中有 {len(absent)} 个不在基因型文件中，'
              f'例如 {", ".join(absent[:5])}')

    todo = [args.only_strain] if args.only_strain else list(args.strains)
    for s in todo:
        print(f'\n[{STRAIN_LABEL.get(s, s)}] {s}  (n_jobs={args.n_jobs})')
        t0 = time.time()
        markers = read_candidates(args.indir, s, args.cand_top_n)
        print(f'  候选 SNP（step1 top {args.cand_top_n}）: {len(markers)}')
        run_strain(s, markers, Xall, grp, args)
        print(f'  用时 {time.time() - t0:.1f} s')

    if args.only_strain:
        return          # 各品系结果已单独落盘，合并由 --merge-only 完成
    merge_outputs(args.strains, args.outdir, args.top_k)


if __name__ == '__main__':
    sys.exit(main())