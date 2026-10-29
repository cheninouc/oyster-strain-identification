#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_rank_integration.py
Step1: 整合 FST 与 PCA 排名，每个品系取 top N 个 SNP。

PCA 改由 PLINK 2.0 计算，本脚本只读其文本输出，不再加载基因型矩阵：

    plink2 --bfile ld/ld_pruned --allow-extra-chr \
           --pca 5 meanimpute allele-wts --out pca/pca

    pca/pca.eigenvec          样本 PC 得分        -> D_k
    pca/pca.eigenvec.allele   每个 SNP 的 PC 权重 -> S_i^PCA

方法（严格对应手稿 2.7 与 Supplementary Methods S1）：
  1. FST 过滤：对 LD 剪枝后的全部位点，取 FST 降序 top 1%。
  2. PCA 打分：
       D_k     = |zbar_T,k - zbar_R,k| / s_pooled,k
       w_k     = D_k / sum(D_k)
       S_i^PCA = sum_k w_k * |l_ik|
  3. 归一化排名（在 FST 过滤后的 N 个位点内）：
       R_i^FST = (N - r_i^FST + 1) / N
       R_i^PCA = (N - r_i^PCA + 1) / N
  4. PCA 过滤：保留 r_i^PCA <= ceil(N * pca_pct)（即 PCA 得分 top 20%）。
  5. 几何平均整合：J_i = sqrt(R_i^FST * R_i^PCA)，按 J 降序取每品系 top N。

关于 l_ik 的尺度：
  PLINK 2.0 的 allele weight 与 scikit-learn 的 PCA.components_ 只相差一个
  “每个 PC 各自独立的”缩放因子（符号按 A1 定义）。手稿公式中的 l_ik 是单位
  方向载荷，因此这里先把每个 PC 的权重向量做 L2 归一化再取绝对值，使其与
  sklearn 实现处于同一尺度，不影响既有结果的解释。
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd

PC_RE = re.compile(r'^PC(\d+)$')


def _pc_columns(names):
    """从列名中挑出 PC1..PCn 并按数值顺序返回。"""
    hits = [(int(PC_RE.match(c).group(1)), c) for c in names if PC_RE.match(c)]
    if not hits:
        raise SystemExit('未找到 PC1..PCn 列，请确认 plink2 --pca 输出是否完整')
    return [c for _, c in sorted(hits)]


def read_eigenvec(path):
    """读 plink2 .eigenvec：表头为 '#IID' 或 '#FID IID'，其后为 PC 列。"""
    df = pd.read_csv(path, sep=r'\s+')
    cols = list(df.columns)
    if 'IID' in cols:                 # '#FID IID PC1 ...'
        id_col = 'IID'
    elif '#IID' in cols:              # '#IID PC1 ...'
        id_col = '#IID'
    elif '#FID' in cols:              # 单列 FID
        id_col = '#FID'
    else:
        raise SystemExit(f'{path}: 首列不是 #IID/#FID，实际列名 {cols[:3]}')

    pc_cols = _pc_columns(cols)
    sample_ids = df[id_col].astype(str).to_numpy()
    U = df[pc_cols].to_numpy(dtype=float)
    return sample_ids, U


def read_allele_wts(path):
    """读 plink2 .eigenvec.allele（或 .eigenvec.var）-> (markers, L[n_pc, n_snp])。

    新版 plink2 默认只输出 #CHROM/ID，没有 POS 列；由于 bed 由
    --set-missing-var-ids '@:#' 生成，ID 本身就是 CHROM:POS，可直接作为位点键。
    """
    head = pd.read_csv(path, sep='\t', nrows=0)
    names = [c.lstrip('#') for c in head.columns]

    pc_cols = _pc_columns(names)
    has_pos = 'POS' in names
    need = ['CHROM'] + (['POS'] if has_pos else []) + ['ID'] + pc_cols
    missing = [c for c in need if c not in names]
    if missing:
        raise SystemExit(f'{path}: 缺少列 {missing}，实际列名 {names}')

    usecols = [head.columns[names.index(c)] for c in need]
    df = pd.read_csv(path, sep='\t', usecols=usecols, low_memory=False)
    df.columns = need

    if has_pos:
        df['marker'] = df['CHROM'].astype(str) + ':' + df['POS'].astype(str)
    else:
        df['marker'] = df['ID'].astype(str)

    # 每个位点按 A1 各出一行（两行权重互为相反数），按位点去重保留首行
    df = df.drop_duplicates(subset='marker', keep='first')

    markers = df['marker'].to_numpy()
    L = df[pc_cols].to_numpy(dtype=float).T          # (n_pc, n_snp)

    # 每个 PC 归一化为单位方向载荷（与 sklearn components_ 一致）
    norm = np.linalg.norm(L, axis=1, keepdims=True)
    L = L / np.where(norm == 0, 1.0, norm)
    return markers, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', required=True, help='sample_pop: 样本名\\t群体')
    ap.add_argument('--eigenvec', required=True, help='plink2 --pca 的 .eigenvec（样本得分）')
    ap.add_argument('--allele-wts', required=True,
                    help='plink2 --pca allele-wts 的 .eigenvec.allele（SNP 权重）')
    ap.add_argument('--strains', nargs='+', required=True, help='7 个选育系')
    ap.add_argument('--fst', nargs='+', required=True, help='FST 文件，顺序与 --strains 对应')
    ap.add_argument('--fst-pct', type=float, default=0.01)
    ap.add_argument('--pca-pct', type=float, default=0.20)
    ap.add_argument('--top-n', type=int, default=2000)
    ap.add_argument('--n-pc', type=int, default=5)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    assert len(args.strains) == len(args.fst), '--strains 与 --fst 数量必须一致'
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 读群体信息 ----------
    pop = pd.read_csv(args.pop, sep=r'\s+', header=None, names=['sample', 'pop'])
    pop_map = dict(zip(pop['sample'].astype(str), pop['pop']))

    # ---------- 读 PCA 结果 ----------
    sample_ids, U = read_eigenvec(args.eigenvec)
    if U.shape[1] < args.n_pc:
        raise SystemExit(f'--n-pc={args.n_pc} 超过 .eigenvec 中的 PC 数({U.shape[1]})')

    markers, L = read_allele_wts(args.allele_wts)
    if L.shape[0] < args.n_pc:
        raise SystemExit(f'--n-pc={args.n_pc} 超过权重文件中的 PC 数({L.shape[0]})')
    n_pc = args.n_pc
    L = L[:n_pc]
    print(f'样本数={len(sample_ids)}  权重位点数={len(markers)}  使用PC数={n_pc}', flush=True)
    print(f'  权重文件前 2 个位点: {markers[:2]}', flush=True)

    # 只保留有群体标签的样本
    known = np.array([s in pop_map for s in sample_ids])
    if (~known).any():
        print(f'  [WARN] {int((~known).sum())} 个样本不在 --pop 中，已从 D_k 计算中剔除')
    U = U[known]
    eig_pops = np.array([pop_map[s] for s in sample_ids[known]])

    marker2col = {m: j for j, m in enumerate(markers)}

    # ---------- 逐品系处理 ----------
    for strain, fstfile in zip(args.strains, args.fst):
        print(f'== strain {strain} ==', flush=True)

        is_t = (eig_pops == strain)
        if is_t.sum() == 0:
            print(f'  [WARN] {strain} 在 .eigenvec 中无样本，跳过')
            continue
        n_t, n_r = int(is_t.sum()), int((~is_t).sum())

        # (a) D_k 与权重 w_k
        D = np.zeros(n_pc)
        for k in range(n_pc):
            zT = U[is_t, k].mean()
            zR = U[~is_t, k].mean()
            sT = U[is_t, k].std(ddof=1)
            sR = U[~is_t, k].std(ddof=1)
            sp = math.sqrt(((n_t - 1) * sT * sT + (n_r - 1) * sR * sR) / (n_t + n_r - 2))
            D[k] = abs(zT - zR) / sp if sp > 0 else 0.0
        w = D / D.sum()
        print(f'  n_t={n_t} n_r={n_r}  D_k={np.round(D, 4)}  w_k={np.round(w, 4)}', flush=True)

        # (b) 读 FST
        fst = pd.read_csv(fstfile, sep=r'\s+', comment='#',
                          names=['CHROM', 'POS', 'FST'])
        fst['marker'] = fst['CHROM'].astype(str) + ':' + fst['POS'].astype(str)
        fst['FST'] = pd.to_numeric(fst['FST'], errors='coerce')
        fst = fst.dropna(subset=['FST']).reset_index(drop=True)

        # FST top 1%
        N_total = len(fst)
        k_fst = max(1, int(math.ceil(N_total * args.fst_pct)))
        fst = fst.sort_values('FST', ascending=False).head(k_fst).reset_index(drop=True)

        # 匹配 PC 权重
        fst['S_pca'] = [
            float(np.sum(w * np.abs(L[:, marker2col[m]])))
            if m in marker2col else np.nan
            for m in fst['marker']
        ]

        n_before = len(fst)
        fst = fst.dropna(subset=['S_pca']).reset_index(drop=True)
        if len(fst) < n_before:
            print(f'  [WARN] {n_before - len(fst)} 个 FST 位点缺 PCA 权重被剔除')
        if len(fst) == 0:
            raise SystemExit(
                'FST 位点与 PCA 权重完全无法匹配，通常是位点键不一致。\n'
                f'  FST 示例: {fstfile}  -> {N_total} 行\n'
                f'  权重示例: {markers[:2].tolist()}\n'
                '  若权重文件不含 POS 列，需保证 bed 的 ID 为 CHROM:POS 形式'
                '（即 --set-missing-var-ids \'@:#\'）；'
                '否则可在 plink2 中加 vcols=chrom,pos,id,... 重新导出。'
            )

        N = len(fst)

        # (c) rank + 归一化
        fst['r_fst'] = fst['FST'].rank(ascending=False, method='min').astype(int)
        fst['r_pca'] = fst['S_pca'].rank(ascending=False, method='min').astype(int)
        fst['R_fst'] = (N - fst['r_fst'] + 1) / N
        fst['R_pca'] = (N - fst['r_pca'] + 1) / N

        # (d) PCA top 20% 过滤
        k_pca = max(1, int(math.ceil(N * args.pca_pct)))
        passed = fst[fst['r_pca'] <= k_pca].copy()

        # (e) 几何平均整合 + top N
        passed['J'] = np.sqrt(passed['R_fst'] * passed['R_pca'])
        passed = passed.sort_values('J', ascending=False).reset_index(drop=True)
        top = passed.head(args.top_n).copy()
        top['strain'] = strain
        top['rank'] = np.arange(1, len(top) + 1)

        out = os.path.join(args.outdir, f'{strain}_top{args.top_n}.tsv')
        cols = ['strain', 'CHROM', 'POS', 'marker', 'FST', 'S_pca',
                'R_fst', 'R_pca', 'J', 'rank']
        top[cols].to_csv(out, sep='\t', index=False)
        print(f'  N_total={N_total}  FST_top1%={N}  PCA20%={k_pca}  '
              f'final_top{args.top_n}={len(top)}  ->  {out}', flush=True)


if __name__ == '__main__':
    main()
