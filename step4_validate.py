#!/usr/bin/env python3
"""step4: 用锁定的 85-SNP panel 在独立验证群体验证七类分类器（对应 SI S1.4）。

训练集 = discovery 群体中 7 个选育系的个体（Wild 不参与训练，因为 panel 只针对
7 个品系优化）；验证集 = step4/ 下已提取好的 125 个个体。两边都只保留 panel SNP。

关键处理：plink --recode A 的 A1 是“每个数据集各自的”等位基因，两个数据集单独
提取时可能选到相反的等位基因（剂量 0/1/2 被翻转）。脚本从 .raw 列名 <ID>_<A1>
读出 A1，凡两侧 A1 不一致的位点统一翻转验证集剂量（2 - x），并把翻转清单写盘。

输出（--outdir）
  predictions.tsv        每个验证个体的真实类别、预测类别与各类后验概率
  confusion_matrix.tsv   计数混淆矩阵（行 = 真实，列 = 预测）
  confusion_rowpct.tsv   按真实类别归一化的混淆矩阵（%）
  metrics.tsv            每类 recall / precision / F1
  allele_alignment.tsv   两侧 A1 不一致、已翻转剂量的位点
  panel_dosages.tsv      验证群体 panel 剂量矩阵（供 R 做 PCA / NJ 树）
  summary.txt            训练与验证的基本信息

用法
  python3 step4_validate.py --panel step3/out/final_panel.tsv \
      --raw-disc panel_disc.raw --pop-disc vcf/sample_pop \
      --raw-val panel_val.raw --pop-val step4/news_sample_pop --outdir step4/out
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASS_ORDER = ['CS', 'JS', 'KS', 'BS', 'GS', 'OS', 'WS']
STRAIN_LABEL = {'CF': 'CS', 'JF': 'JS', 'KF': 'KS', 'SB': 'BS',
                'SG': 'GS', 'SO': 'OS', 'SW': 'WS'}
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ---------------------------------------------------------------- 位点标识
def chrom_num(c):
    """把 NC_047559.1 / Chr9 / 9 统一成 '9'，便于跨文件比对位点。"""
    c = c.strip()
    m = re.fullmatch(r'NC_047(\d{3})\.\d+', c)
    if m:
        return str(int(m.group(1)) - 558)
    m = re.fullmatch(r'[Cc]hr(?:omosome)?_?(\d+)', c)
    if m:
        return str(int(m.group(1)))
    m = re.fullmatch(r'(\d+)', c)
    if m:
        return str(int(m.group(1)))
    return c


def canon(name):
    """<CHROM>:<POS>[:REF:ALT][_A1] -> (chrom, pos)。用于匹配两套 .raw 的列名。"""
    base = re.sub(r'_[A-Za-z<>]+$', '', name)      # 去掉 --recode A 追加的 _A1
    parts = base.split(':')
    if len(parts) >= 2 and parts[1].isdigit():
        return (chrom_num(parts[0]), int(parts[1]))
    return (base, -1)


def a1_of(colname):
    """从 .raw 列名 <ID>_<A1> 取 A1；没有后缀时返回 ''。"""
    m = re.search(r'_([A-Za-z<>]+)$', colname)
    return m.group(1) if m else ''


# ---------------------------------------------------------------- 输入
def read_pop(path):
    pop = pd.read_csv(path, sep=r'\s+', header=None,
                      names=['IID', 'POP'], dtype=str)
    pop = pop.dropna(subset=['IID'])
    dup = int(pop['IID'].duplicated().sum())
    if dup:
        pop = pop.drop_duplicates('IID', keep='first')
        print(f'  警告: {path} 有 {dup} 个重复 IID，已保留首条')
    return pop


def read_raw(path, panel, tag):
    """读 plink --recode A 的 .raw，只保留 panel SNP。

    返回 (X, ids, a1)。X 的列名统一为 panel 里的 marker 写法，列序与 panel 一致。
    """
    df = pd.read_csv(path, sep=r'\s+', engine='python', dtype=str)
    if df.shape[1] < 7:
        raise SystemExit(f'{path}: 列数 {df.shape[1]}，不是 --recode A 的 .raw 文件')
    ids = df.iloc[:, :2].copy()
    ids.columns = ['FID', 'IID']

    key2col, key2a1 = {}, {}
    for c in df.columns[6:]:
        k = canon(c)
        if k not in key2col:                 # 同一位点重复出现时保留首列
            key2col[k] = c
            key2a1[k] = a1_of(c)

    keys = [canon(m) for m in panel]
    missing = [m for m, k in zip(panel, keys) if k not in key2col]
    if missing:
        raise SystemExit(f'{tag}: 缺少 {len(missing)} 个 panel SNP，例如 {missing[:3]}\n'
                         f'  实际前 3 列: {list(df.columns[6:9])}')
    if len(missing) == 0 and len(panel) != len(set(keys)):
        print(f'  警告: panel 中存在同位点重复（去重后 {len(set(keys))} 个）')

    X = pd.DataFrame({m: pd.to_numeric(df[key2col[k]], errors='coerce')
                      for m, k in zip(panel, keys)}, columns=panel)
    a1 = {m: key2a1[k] for m, k in zip(panel, keys)}
    print(f'  {tag}: {len(ids)} 个体 × {X.shape[1]} SNP（.raw {df.shape[1] - 6} 列）')
    return X.reset_index(drop=True), ids.reset_index(drop=True), a1, missing


def align_samples(ids, pop, tag):
    """把 sample_pop 的群体标签对齐到 .raw 的样本行（多键回退）。"""
    known = dict(zip(pop['IID'].astype(str), pop['POP'].astype(str)))
    out, via, used = [], {'IID': 0, 'FID_IID': 0, 'hyphen': 0, 'unmatched': 0}, []
    for fid, iid in zip(ids['FID'].astype(str), ids['IID'].astype(str)):
        if iid in known:
            out.append(known[iid]); used.append(iid); via['IID'] += 1; continue
        hit, path, key = None, None, None
        if fid not in ('0', '', 'nan', iid):
            for k in (f'{fid}_{iid}', f'{fid}-{iid}'):
                if k in known:
                    hit, path, key = known[k], 'FID_IID', k
                    break
        if hit is None:
            for k in (iid.replace('-', '_'), iid.replace('_', '-')):
                if k in known:
                    hit, path, key = known[k], 'hyphen', k
                    break
        if hit is None:
            out.append(np.nan); used.append(None); via['unmatched'] += 1
        else:
            out.append(hit); used.append(key); via[path] += 1
    lab = pd.Series(out, dtype=object)
    lab.name = 'strain'
    print(f'  {tag}: 标签匹配 IID {via["IID"]} / FID 拼接 {via["FID_IID"]} / '
          f'分隔符回退 {via["hyphen"]} / 未匹配 {via["unmatched"]}')
    if via['unmatched']:
        miss = [u for u in used if u is None][:5]
        print(f'    未匹配示例: {miss}（将被剔除）')
    return lab.map(lambda g: STRAIN_LABEL.get(g, g))


def keep_strains(X, lab, ids, tag):
    keep = lab.isin(CLASS_ORDER).values
    X = X.loc[keep].reset_index(drop=True)
    lab = lab.loc[keep].reset_index(drop=True)
    ids = ids.loc[keep].reset_index(drop=True)
    if len(X) == 0:
        raise SystemExit(f'{tag}: 没有可用个体（标签需为 {CLASS_ORDER} 或 CF/JF/KF/SB/SG/SO/SW）')
    n = lab.value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    print(f'    {tag} 保留 {len(X)} 个体: ' + ', '.join(f'{k}={n[k]}' for k in CLASS_ORDER))
    return X, lab, ids, n


def align_coding(Xd, ad, Xv, av):
    """两侧 A1 不一致时翻转验证集剂量（2 - x），返回翻转清单。"""
    flipped = []
    for m in Xd.columns:
        a, b = ad.get(m, ''), av.get(m, '')
        if a and b and a.upper() != b.upper():
            Xv[m] = 2.0 - Xv[m]
            flipped.append((m, a, b))
    return Xv, flipped


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--raw-disc', required=True)
    ap.add_argument('--pop-disc', required=True)
    ap.add_argument('--raw-val', required=True)
    ap.add_argument('--pop-val', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=20261027)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    panel_df = pd.read_csv(args.panel, sep='\t')
    if 'marker' not in panel_df.columns:
        raise SystemExit(f'{args.panel}: 缺少 marker 列')
    if 'rank' in panel_df.columns:
        panel_df = panel_df.sort_values('rank')
    panel = panel_df['marker'].astype(str).tolist()
    print(f'锁定 panel: {len(panel)} 个 SNP')

    print('[1/4] 读取基因型')
    Xd, idd, ad, _ = read_raw(args.raw_disc, panel, 'discovery')
    Xv, idv, av, _ = read_raw(args.raw_val, panel, 'validation')

    print('[2/4] 对齐群体标签')
    labd = align_samples(idd, read_pop(args.pop_disc), 'discovery')
    labv = align_samples(idv, read_pop(args.pop_val), 'validation')
    Xd, yd, idd, n_disc = keep_strains(Xd, labd, idd, 'discovery')
    Xv, yv, idv, n_val = keep_strains(Xv, labv, idv, 'validation')
    if int((n_disc > 0).sum()) < 2:
        raise SystemExit('discovery 可训练类别不足 2 个：请检查 --pop-disc 的群体代码'
                         '（CF/JF/KF/SB/SG/SO/SW 或 CS/JS/KS/BS/GS/OS/WS）是否匹配')

    Xv, flipped = align_coding(Xd, ad, Xv, av)
    if flipped:
        print(f'[3/4] 等位基因方向: {len(flipped)} 个位点两侧 A1 相反，已翻转验证集剂量')
    else:
        print('[3/4] 等位基因方向: 全部一致，无需翻转')

    print('[4/4] 训练并预测')
    model = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', C=1.0,
                                   class_weight='balanced', max_iter=10000,
                                   random_state=args.seed)),
    ])
    model.fit(Xd.values, yd.values)
    classes = list(model.named_steps['clf'].classes_)
    proba = model.predict_proba(Xv.values)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    print(f'  训练 {len(Xd)} 个体 | 输出类别 {classes} | 验证 {len(yv)} 个体')

    # ---- 写出 ----
    pred_df = pd.DataFrame({'FID': idv['FID'], 'IID': idv['IID'],
                            'true_strain': yv.values, 'predicted_strain': pred,
                            'max_posterior': proba.max(axis=1).round(6)})
    for j, c in enumerate(classes):
        pred_df[f'P_{c}'] = proba[:, j].round(6)
    pred_df.to_csv(os.path.join(args.outdir, 'predictions.tsv'), sep='\t', index=False)

    cm_raw = confusion_matrix(yv, pred, labels=CLASS_ORDER)
    cm = pd.DataFrame(cm_raw, index=CLASS_ORDER, columns=CLASS_ORDER)
    cm.index.name = 'true\\pred'
    cm.to_csv(os.path.join(args.outdir, 'confusion_matrix.tsv'), sep='\t')
    row = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100
    row.round(2).to_csv(os.path.join(args.outdir, 'confusion_rowpct.tsv'), sep='\t')

    rec = recall_score(yv, pred, labels=CLASS_ORDER, average=None, zero_division=0)
    pre = precision_score(yv, pred, labels=CLASS_ORDER, average=None, zero_division=0)
    f1 = f1_score(yv, pred, labels=CLASS_ORDER, average=None, zero_division=0)
    pd.DataFrame({
        'strain': CLASS_ORDER,
        'n': [int(n_val[c]) for c in CLASS_ORDER],
        'recall': np.round(rec, 4),
        'precision': np.round(pre, 4),
        'f1': np.round(f1, 4),
    }).to_csv(os.path.join(args.outdir, 'metrics.tsv'), sep='\t', index=False)

    if flipped:
        pd.DataFrame(flipped, columns=['marker', 'A1_discovery',
                                       'A1_validation']).to_csv(
            os.path.join(args.outdir, 'allele_alignment.tsv'), sep='\t', index=False)

    dos = Xv.copy()
    dos.insert(0, 'strain', yv.values)
    dos.insert(1, 'IID', idv['IID'].values)
    dos.to_csv(os.path.join(args.outdir, 'panel_dosages.tsv'), sep='\t', index=False)

    correct = int((np.asarray(yv) == pred).sum())
    acc = accuracy_score(yv, pred)
    bacc = balanced_accuracy_score(yv, pred)
    mf1 = f1_score(yv, pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(yv, pred)
    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as fh:
        fh.write(f'panel_snps\t{len(panel)}\n')
        fh.write(f'classes\t{len(CLASS_ORDER)}\n')
        fh.write('classifier\tmultinomial_L2_lbfgs_C1.0_balanced\n')
        fh.write(f'discovery_n\t{len(Xd)}\n')
        fh.write(f'validation_n\t{len(yv)}\n')
        fh.write(f'correct\t{correct}\n')
        fh.write(f'accuracy\t{acc:.4f}\n')
        fh.write(f'balanced_accuracy\t{bacc:.4f}\n')
        fh.write(f'macro_f1\t{mf1:.4f}\n')
        fh.write(f'mcc\t{mcc:.4f}\n')
        fh.write(f'min_class_recall\t{float(np.min(rec)):.4f}\n')
        fh.write(f'allele_flips\t{len(flipped)}\n')

    print(f'  正确 {correct}/{len(yv)} | accuracy {acc:.4f} | balanced acc '
          f'{bacc:.4f} | macro-F1 {mf1:.4f} | MCC {mcc:.4f} | '
          f'min recall {float(np.min(rec)):.4f}')
    print(f'完成。结果在: {os.path.abspath(args.outdir)}')


if __name__ == '__main__':
    main()
