#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# step4: 用锁定的 85-SNP panel 在 125 个独立验证个体上验证（SI S1.4）
#
# 输入（已就位）
#   step3/out/final_panel.tsv   step3 锁定的 panel
#   step4/snp_subset.vcf        125 个体 × 85 SNP 的验证基因型
#   step4/news_sample_pop       验证个体 -> 群体（两列：IID POP）
#   vcf/sample_pop              discovery 群体标签
#   step1/ld/ld_pruned          discovery 的 LD 剪枝 bed
#
# 输出
#   step4/out/{predictions,confusion_matrix,confusion_rowpct,metrics,
#              allele_alignment,panel_dosages}.tsv 与 summary.txt
#
# 依赖: plink (1.9, --recode A)、python3 (numpy, pandas, scikit-learn)
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

## ---------- 配置 ----------
WORKDIR="/public1/node3_chcg/zhiwen/new/new_robot"
STEP1="$WORKDIR/step1"
STEP3="$WORKDIR/step3"
STEP4="$WORKDIR/step4"
OUTDIR="$STEP4/out"

PANEL="$STEP3/out/final_panel.tsv"
VAL_VCF="$STEP4/snp_subset.vcf"
POP_DISC="$WORKDIR/vcf/sample_pop"
POP_VAL="$STEP4/news_sample_pop"

## ---------- 工具与输入检测 ----------
PLINK=$(command -v plink || true)
[ -z "$PLINK" ] && { echo "ERROR: plink(1.9) 未找到"; exit 1; }
python3 -c "import sklearn, pandas, numpy" 2>/dev/null || {
  echo "ERROR: 缺少 python 依赖（scikit-learn / pandas / numpy）"; exit 1; }
for f in "$PANEL" "$VAL_VCF" "$POP_DISC" "$POP_VAL"; do
  [ -f "$f" ] || { echo "ERROR: 找不到 $f"; exit 1; }
done
[ -f "$STEP1/ld/ld_pruned.bed" ] || {
  echo "ERROR: 找不到 $STEP1/ld/ld_pruned.bed（discovery 基因型）"; exit 1; }

mkdir -p "$OUTDIR"
cd "$OUTDIR"

## ---------- 1. panel SNP 列表 ----------
echo "[1/4] 写出 panel SNP 列表"
python3 - "$PANEL" > panel.txt <<'EOF'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1], sep='\t')
if 'rank' in d.columns:
    d = d.sort_values('rank')
for m in d['marker'].astype(str):
    print(m)
EOF
echo "  $(wc -l < panel.txt) 个 SNP"

## ---------- 2. discovery 基因型 ----------
## --keep-allele-order：保持 .bim 的 A1/A2 顺序，避免按“本次样本的次等位基因”
## 重排；否则两边单独提取时剂量 (0/1/2) 可能整体翻转，脚本虽会兜底修正，
## 但保持顺序能让 .raw 列名与 panel 完全一致。
echo "[2/4] 提取 discovery 基因型"
plink --bfile "$STEP1/ld/ld_pruned" --allow-extra-chr --keep-allele-order \
  --extract panel.txt --recode A --out panel_disc > /dev/null

## ---------- 3. 验证群体基因型 ----------
## --double-id：让 FID=IID=原始样本名，避免按首个 '_' 拆名导致对不上标签
echo "[3/4] 提取验证群体基因型"
plink --vcf "$VAL_VCF" --allow-extra-chr --double-id --keep-allele-order \
  --vcf-half-call m --set-missing-var-ids '@:#' \
  --extract panel.txt --recode A --out panel_val > /dev/null

## ---------- 4. 训练、预测、评估 ----------
echo "[4/4] 训练七类分类器并在验证群体上评估"
python3 "$SCRIPT_DIR/step4_validate.py" \
  --panel   "$PANEL" \
  --raw-disc panel_disc.raw --pop-disc "$POP_DISC" \
  --raw-val  panel_val.raw  --pop-val  "$POP_VAL" \
  --outdir "$OUTDIR"

echo ""
echo "完成。结果在: $OUTDIR/"
echo "  predictions.tsv        逐个体预测与后验概率"
echo "  confusion_matrix.tsv   计数混淆矩阵（Fig 3F / Table S4）"
echo "  metrics.tsv            每类 recall / precision / F1"
echo "  allele_alignment.tsv   两侧 A1 相反、已翻转剂量的位点清单"
echo "  panel_dosages.tsv      验证群体 85-SNP 剂量矩阵（供 R 画 PCA 与 NJ 树）"
echo "  summary.txt            训练/验证摘要"
