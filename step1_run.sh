#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# step1: LD 剪枝 -> FST(top1%, target-vs-rest) + PCA(PLINK2 allele-wts)
#        -> FST/PCA 归一化排名整合 -> 每品系 top N (默认 2000)
#
# 依赖: plink  (1.9, 用于 VCF->bed 与 --indep-pairwise)
#       plink2 (2.0, 用于 --pca 并输出 SNP 权重 allele-wts)
#       vcftools / python3 (numpy, pandas)
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

## ---------- 配置（按需修改） ----------
WORKDIR="/public1/node3_chcg/zhiwen/new/new_robot"
VCF="$WORKDIR/vcf/all_snp.recode.vcf.gz"
POP="$WORKDIR/vcf/sample_pop"
OUTDIR="$WORKDIR/step1"

STRAINS=(CF JF KF SB SG SO SW)      # 7 个选育系（对应 CS/JS/KS/BS/GS/OS/WS）
                                     # rest 组 = 其余 6 品系 + Wild

LD_WINDOW=50    # --indep-pairwise 窗口 (SNP 数, PLINK1.9 语义)
LD_STEP=5       # 步长 (SNP 数)
LD_R2=0.5       # r2 阈值

FST_PCT=0.01        # FST top 1%
PCA_PCT=0.20        # PCA top 20%
TOP_N=2000          # 每品系保留 SNP 数
N_PC=5              # 前 5 个主成分
THREADS=8           # plink2 线程数

## ---------- 工具检测 ----------
PLINK=$(command -v plink || true)
PLINK2=$(command -v plink2 || true)
VCFTOOLS=$(command -v vcftools || true)
if [ -z "$PLINK" ]; then echo "ERROR: plink(1.9) 未找到"; exit 1; fi
if [ -z "$PLINK2" ]; then echo "ERROR: plink2 未找到（--pca allele-wts 需要 PLINK 2.0）"; exit 1; fi
if [ -z "$VCFTOOLS" ]; then echo "ERROR: vcftools 未找到"; exit 1; fi

mkdir -p "$OUTDIR"/{ld,fst,pop,pca,out}
cd "$OUTDIR"

## ---------- 1. 分组文件 (target / rest) ----------
echo "[1/6] 生成 target/rest 分组文件"
for S in "${STRAINS[@]}"; do
  awk -v p="$S" '$2==p{print $1}' "$POP" > pop/${S}.txt
  awk -v p="$S" '$2!=p{print $1}' "$POP" > pop/rest_${S}.txt
  echo "  $S: target=$(wc -l < pop/${S}.txt)  rest=$(wc -l < pop/rest_${S}.txt)"
done

## ---------- 2. VCF -> bed ----------
echo "[2/6] VCF 转 bed（保留 CHROM，ID 唯一化为 CHROM:POS）"
plink --vcf "$VCF" --allow-extra-chr \
  --set-missing-var-ids '@:#' --make-bed --out ld/all_snp

## ---------- 3. LD 剪枝 ----------
echo "[3/6] LD 剪枝 --indep-pairwise $LD_WINDOW $LD_STEP $LD_R2"
plink --bfile ld/all_snp --allow-extra-chr \
  --indep-pairwise $LD_WINDOW $LD_STEP $LD_R2 --out ld/ld
echo "  LD 剪枝后独立 SNP 数: $(wc -l < ld/ld.prune.in)"

plink --bfile ld/all_snp --extract ld/ld.prune.in \
  --allow-extra-chr --make-bed --out ld/ld_pruned

## ---------- 4. PCA (PLINK2, 同时输出 SNP 权重) ----------
# 注意: 不要加 --maf，否则 PCA 位点集会与已完成的 FST 位点集不同步
echo "[4/6] PLINK2 PCA --pca $N_PC meanimpute allele-wts"
plink2 --bfile ld/ld_pruned --allow-extra-chr \
  --pca "$N_PC" meanimpute allele-wts \
  --threads "$THREADS" --out pca/pca
echo "  样本得分: $(wc -l < pca/pca.eigenvec) 行（含表头）"
echo "  SNP 权重: $(wc -l < pca/pca.eigenvec.allele) 行（含表头）"

## ---------- 5. FST (每品系 target-vs-rest) ----------
echo "[5/6] VCFtools FST (target-vs-rest)"
awk 'BEGIN{OFS="\t"} {print $1, $4}' ld/ld_pruned.bim > ld/ld_pruned.positions
for S in "${STRAINS[@]}"; do
  vcftools --gzvcf "$VCF" \
    --positions ld/ld_pruned.positions \
    --weir-fst-pop pop/${S}.txt \
    --weir-fst-pop pop/rest_${S}.txt \
    --out fst/fst_${S} > /dev/null 2>&1 || true
  n=$(grep -vc '^#' fst/fst_${S}.weir.fst 2>/dev/null || echo 0)
  echo "  $S: FST 位点数=$n"
done

## ---------- 6. Python 排名整合 ----------
echo "[6/6] FST/PCA 归一化排名整合 -> 每品系 top $TOP_N"
FSTFILES=""
for S in "${STRAINS[@]}"; do
  FSTFILES="$FSTFILES fst/fst_${S}.weir.fst"
done

python3 "$SCRIPT_DIR/step1_rank_integration.py" \
  --pop "$POP" \
  --eigenvec pca/pca.eigenvec \
  --allele-wts pca/pca.eigenvec.allele \
  --strains "${STRAINS[@]}" \
  --fst $FSTFILES \
  --fst-pct "$FST_PCT" \
  --pca-pct "$PCA_PCT" \
  --top-n "$TOP_N" \
  --n-pc "$N_PC" \
  --outdir out

echo ""
echo "完成。每品系 top ${TOP_N} 结果在: $OUTDIR/out/"
