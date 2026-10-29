#!/usr/bin/env python3
"""step3：Multinomial Elastic Net 稳定性排序 + top-N panel 大小优化。

严格按 Supplementary Methods S1.3 实现：

  1) 在 1,400 个候选 SNP 上做多分类 Elastic Net（saga，balanced），
     C ∈ {0.01, 0.03, 0.1, 0.3, 1.0} × l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}，
     分层 5 折、以 macro-F1 选超参；
  2) 100 次分层子采样（每次 80% 个体），某 SNP 在任一类上的系数非零即“被选中”，
     系数幅度取跨类系数的 L2 范数；
  3) 排序：选择频率降序 → 被选中迭代的 mean L2 降序 → 全部迭代的 mean L2 降序；
  4) 沿固定排序构造累积 top-N panel，用固定的多分类 L2 逻辑回归
     （lbfgs, C=1.0, balanced, max_iter=10000，中位数填补 + 标准化）做
     重复分层 5 折（10 次重复 = 50 个测试折）评估；
  5) 最终 panel = 四个指标（macro-F1、balanced accuracy、MCC、最小类召回）
     均值全部达到目标值的最小 N。

类别数由 --classes 控制：
  7 = 只用 7 个选育品系（Wild 作为背景样本，仅参与 step2 的 rest 集）
  8 = 7 个选育品系 + Wild 作为独立类别（原 S1.3 设定）

用法：
  python3 step3_elasticnet.py --geno geno.raw --pop sample_pop \
      --candidates candidates_union.tsv --classes 7 \
      --ranking-out ranking.tsv --panel-out final_panel.tsv
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, recall_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2_elasticnet import (align_pop, build_pipeline, read_pop,  # noqa: E402
                              read_raw)

SK_VER = tuple(int(v) for v in sklearn.__version__.split('.')[:2])
NZ_TOL = 1e-8
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
STRAIN_LABEL = {'CF': 'CS', 'JF': 'JS', 'KF': 'KS', 'SB': 'BS',
                'SG': 'GS', 'SO': 'OS', 'SW': 'WS'}
STRAINS = list(STRAIN_LABEL)

warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------- 数据
def build_dataset(args):
    """读 .raw + sample_pop，取候选 SNP，构造 X / y。"""
    X, raw_ids = read_raw(args.geno)
    pop = read_pop(args.pop)
    grp, via, _ = align_pop(raw_ids, pop)

    cand = [m for m in pd.read_csv(args.candidates, sep='\t')['marker']
            .astype(str)]
    missing = [m for m in cand if m not in X.columns]
    if missing:
        raise SystemExit(f'{len(missing)} 个候选 SNP 不在基因型文件中，'
                         f'例如 {missing[:5]}')
    X = X[cand].reset_index(drop=True)

    lab = grp.map(lambda g: STRAIN_LABEL.get(g, g))
    valid = list(STRAIN_LABEL.values())          # CS/JS/KS/BS/GS/OS/WS
    if args.classes == 8:
        valid = valid + ['Wild']
    keep = lab.isin(valid)
    X, lab = X[keep.values].reset_index(drop=True), lab[keep.values].reset_index(drop=True)
    y = lab.values

    n_wild = int((lab == 'Wild').sum())
    print(f'基因型矩阵: {X.shape[0]} 样本 × {X.shape[1]} SNP')
    print(f'类别数: {args.classes} | Wild 样本 {n_wild} 个'
          + ('（已排除）' if args.classes == 7 else '（作为独立类别）'))
    print('样本对齐: ' + '，'.join(f'{k} {v}' for k, v in via.items() if k != 'unmatched')
          + f'，未匹配 {via["unmatched"]}')
    print('类别样本数: ' + '，'.join(f'{c} {int((y == c).sum())}'
                                 for c in sorted(set(y))))
    return X, y


def stratified_folds(y, n_splits, seed):
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed).split(np.zeros(len(y)), y))


# ---------------------------------------------------------------- 超参数
def fit_score(X, y, tr, te, C, l1, seed):
    pipe = build_pipeline(C, l1, seed)
    pipe.fit(X.iloc[tr], y[tr])
    return f1_score(y[te], pipe.predict(X.iloc[te]), average='macro',
                    zero_division=0)


def _combo_score(X, y, folds, C, l, seed):
    return C, l, float(np.mean([fit_score(X, y, tr, te, C, l, seed)
                                for tr, te in folds]))


def tune_hyperparams(X, y, folds, seed, n_jobs):
    """C × l1 网格，分层 5 折，macro-F1 均值最大者胜。"""
    grid = [(C, l) for C in C_GRID for l in L1_GRID]
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_combo_score)(X, y, folds, C, l, seed) for C, l in grid)
    tab = pd.DataFrame(rows, columns=['C', 'l1_ratio', 'mean_macro_f1'])
    tab = tab.sort_values(['mean_macro_f1', 'l1_ratio', 'C'],
                          ascending=[False, False, True]).reset_index(drop=True)
    best = tab.iloc[0]
    print(f'最优超参: C={best.C:g}  l1_ratio={best.l1_ratio:g}  '
          f'mean macro-F1={best.mean_macro_f1:.4f}')
    return float(best.C), float(best.l1_ratio), tab


# ---------------------------------------------------------------- 稳定性排序
def _subsample_fit(X, y, tr, C, l1, seed):
    pipe = build_pipeline(C, l1, seed)
    pipe.fit(X.iloc[tr], y[tr])
    coef = pipe.named_steps['clf'].coef_          # (n_classes, n_features)
    sel = (np.abs(coef) > NZ_TOL).any(axis=0)
    l2 = np.sqrt((coef ** 2).sum(axis=0))
    return sel, l2


def stability_ranking(X, y, C, l1, n_sub, frac, seed, n_jobs):
    sss = StratifiedShuffleSplit(n_splits=n_sub, train_size=frac, random_state=seed)
    tasks = [tr for tr, _ in sss.split(np.zeros(len(y)), y)]
    print(f'稳定性分析: {n_sub} 次分层子采样 × {frac:.0%} 个体 ...')
    res = Parallel(n_jobs=n_jobs)(
        delayed(_subsample_fit)(X, y, tr, C, l1, seed + i)
        for i, tr in enumerate(tasks))

    sel = np.array([r[0] for r in res], dtype=float)      # (n_sub, n_snp)
    l2 = np.array([r[1] for r in res], dtype=float)
    freq = sel.mean(axis=0)
    n_sel = sel.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_l2_sel = np.where(n_sel > 0, (l2 * sel).sum(axis=0) / n_sel, 0.0)
    mean_l2_all = l2.mean(axis=0)

    tab = pd.DataFrame({'marker': X.columns, 'selection_frequency': freq,
                        'mean_l2_selected': mean_l2_sel,
                        'mean_l2_all': mean_l2_all})
    tab = tab.sort_values(['selection_frequency', 'mean_l2_selected', 'mean_l2_all'],
                          ascending=False).reset_index(drop=True)
    tab['rank'] = np.arange(1, len(tab) + 1)
    nz = int((tab.selection_frequency > 0).sum())
    print(f'  曾被选中的候选: {nz}/{len(tab)} | 频率=1.00 的: '
          f'{int((tab.selection_frequency >= 0.999).sum())}')
    return tab[['rank', 'marker', 'selection_frequency',
                'mean_l2_selected', 'mean_l2_all']]


# ---------------------------------------------------------------- panel 评估
def build_eval_model(seed):
    """固定的多分类 L2 逻辑回归（S1.3：lbfgs, C=1.0, balanced）。"""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', C=1.0,
                                   class_weight='balanced',
                                   max_iter=10000, random_state=seed)),
    ])


def eval_panel(N, markers, X, y, folds, seed):
    """在 N 个 SNP 上做一轮 5 折评估，返回各项指标与逐类召回。"""
    Xs = X[markers[:N]]
    rec, mcc, ba, f1 = [], [], [], []
    class_rec = {}
    for tr, te in folds:
        pipe = build_eval_model(seed)
        pipe.fit(Xs.iloc[tr], y[tr])
        pred = pipe.predict(Xs.iloc[te])
        yt = y[te]
        f1.append(f1_score(yt, pred, average='macro', zero_division=0))
        ba.append(balanced_accuracy_score(yt, pred))
        mcc.append(matthews_corrcoef(yt, pred))
        labels = sorted(set(yt))
        r = recall_score(yt, pred, labels=labels, average=None, zero_division=0)
        rec.append(float(np.min(r)))
        for c, v in zip(labels, r):
            class_rec.setdefault(c, []).append(float(v))
    return {'n_snps': N, 'macro_f1': float(np.mean(f1)),
            'macro_f1_sd': float(np.std(f1, ddof=1)),
            'balanced_accuracy': float(np.mean(ba)),
            'balanced_accuracy_sd': float(np.std(ba, ddof=1)),
            'mcc': float(np.mean(mcc)), 'mcc_sd': float(np.std(mcc, ddof=1)),
            'min_recall': float(np.mean(rec)),
            'min_recall_sd': float(np.std(rec, ddof=1)),
            'min_recall_fold_min': float(np.min(rec)),
            'per_class': {c: float(np.mean(v)) for c, v in class_rec.items()}}


def eval_panels(sizes, markers, X, y, folds, seed, n_jobs):
    out = Parallel(n_jobs=n_jobs)(
        delayed(eval_panel)(N, markers, X, y, folds, seed) for N in sizes)
    tab = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_class'}
                        for r in out])
    per = pd.DataFrame([r['per_class'] for r in out], index=sizes).T
    per.index.name = 'class'
    per.columns.name = 'n_snps'
    return tab.sort_values('n_snps').reset_index(drop=True), per


def first_pass(tab, target):
    """返回四个指标均值全部达标的最小 N（不存在则 None）。"""
    ok = (tab.macro_f1 >= target) & (tab.balanced_accuracy >= target) & \
         (tab.mcc >= target) & (tab.min_recall >= target)
    if not ok.any():
        return None
    return int(tab.loc[ok, 'n_snps'].min())


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(
        description='step3: multinomial Elastic Net 稳定性排序 + panel 优化')
    p.add_argument('--geno', required=True, help='plink --recode A 的 .raw')
    p.add_argument('--pop', required=True, help='sample_pop（两列：样本名 群体）')
    p.add_argument('--candidates', required=True, help='step2 的 candidates_union.tsv')
    p.add_argument('--classes', type=int, choices=[7, 8], default=7,
                   help='7 = 七个选育品系；8 = 另加 Wild 作为独立类别（默认 7）')
    p.add_argument('--n-sub', type=int, default=100)
    p.add_argument('--sub-frac', type=float, default=0.80)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--coarse-step', type=int, default=10)
    p.add_argument('--coarse-max', type=int, default=200)
    p.add_argument('--refine', type=int, default=6,
                   help='在转折点前后 ±N 个 SNP 内做单 SNP 分辨率评估')
    p.add_argument('--target', type=float, default=1.0,
                   help='四个指标必须达到的目标值（默认 1.0）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-jobs', type=int, default=-1)
    p.add_argument('--outdir', required=True)
    p.add_argument('--tag', default='', help='输出文件名后缀，如 _7class')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    X, y = build_dataset(args)

    # 1) 超参
    folds_tune = stratified_folds(y, args.folds, args.seed)
    C, l1, hp = tune_hyperparams(X, y, folds_tune, args.seed, args.n_jobs)
    hp.insert(0, 'classes', args.classes)
    hp.to_csv(os.path.join(args.outdir, f'hyperparameters{args.tag}.tsv'),
              sep='\t', index=False)

    # 2) 稳定性排序
    rank = stability_ranking(X, y, C, l1, args.n_sub, args.sub_frac,
                             args.seed, args.n_jobs)
    rank.to_csv(os.path.join(args.outdir, f'ranking{args.tag}.tsv'),
                sep='\t', index=False)
    markers = rank.sort_values('rank')['marker'].tolist()

    # 3) 固定 10 次重复的 CV 折（所有 N 共用，保证可比）
    rng = np.random.RandomState(args.seed)
    folds = []
    for s in rng.randint(0, 10 ** 6, args.repeats):
        folds.extend(stratified_folds(y, args.folds, int(s)))

    coarse = list(range(args.coarse_step,
                        min(args.coarse_max, len(markers)) + 1, args.coarse_step))
    print(f'panel 评估: 粗筛 {len(coarse)} 个规模 × {args.repeats * args.folds} 折 ...')
    tab_c, per_c = eval_panels(coarse, markers, X, y, folds, args.seed, args.n_jobs)

    n0 = first_pass(tab_c, args.target)
    tab, per = tab_c, per_c
    if n0 is not None and args.refine > 0:
        fine = [n for n in range(max(args.folds, n0 - args.refine),
                                 min(n0 + args.refine, len(markers)) + 1)
                if n not in set(coarse)]
        if fine:
            print(f'单 SNP 分辨率细化: {fine[0]}–{fine[-1]} ({len(fine)} 个规模) ...')
            tab_f, per_f = eval_panels(fine, markers, X, y, folds,
                                       args.seed, args.n_jobs)
            tab = pd.concat([tab, tab_f]).sort_values('n_snps').reset_index(drop=True)
            per = per.join(per_f, how='outer').sort_index(axis=1)
            per = per[[c for c in tab.n_snps if c in per.columns]]

    tab.to_csv(os.path.join(args.outdir, f'panel_size_metrics{args.tag}.tsv'),
               sep='\t', index=False)
    per.to_csv(os.path.join(args.outdir, f'panel_class_recall{args.tag}.tsv'),
               sep='\t')

    # 4) 最终 panel
    n_final = first_pass(tab, args.target)
    if n_final is None:
        best = tab.sort_values(['min_recall', 'macro_f1', 'mcc'],
                               ascending=False).iloc[0]
        print(f'\n!! 没有规模同时达到 target={args.target}；'
              f'最小类召回最高的规模为 {int(best.n_snps)}'
              f'（macro-F1 {best.macro_f1:.3f}, min recall {best.min_recall:.3f}）')
        n_final = int(best.n_snps)
    else:
        row = tab[tab.n_snps == n_final].iloc[0]
        print(f'\n最终 panel: {n_final} SNPs | macro-F1 {row.macro_f1:.3f} | '
              f'balanced acc {row.balanced_accuracy:.3f} | MCC {row.mcc:.3f} | '
              f'min recall {row.min_recall:.3f}')

    final = rank[rank['rank'] <= n_final].copy()
    final.to_csv(os.path.join(args.outdir, f'final_panel{args.tag}.tsv'),
                 sep='\t', index=False)
    with open(os.path.join(args.outdir, f'summary{args.tag}.txt'), 'w') as fh:
        fh.write(f'classes\t{args.classes}\n'
                 f'C\t{C:g}\nl1_ratio\t{l1:g}\n'
                 f'final_panel_size\t{n_final}\n'
                 f'runtime_min\t{(time.time() - t0) / 60:.1f}\n')
    print(f'输出目录: {args.outdir} | 用时 {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
