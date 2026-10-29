#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: 本脚本使用了 bash 数组，请用 bash 运行："
  echo "  bash $0"
  exit 1
fi

# ============================================================
# step3: Multinomial Elastic Net 稳定性排序 + panel 大小优化（S1.3）
#        承接 step2 的 1,400 个候选 SNP
#
# 依赖: python3 (numpy, pandas, scikit-learn, joblib)
#       plink (1.9) —— 仅当 step2 的 geno.raw 不存在时用于重建
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

## ---------- 配置（按需修改） ----------
WORKDIR="/public1/node3_chcg/zhiwen/new/new_robot"
STEP2="$WORKDIR/step2"
STEP2_OUT="$STEP2/out"
POP="$WORKDIR/vcf/sample_pop"
OUTDIR="$WORKDIR/step3"

CLASSES=7            # 7 = 七个选育品系（Wild 仅作背景，不参赛）
                     # 8 = 加上 Wild 作为独立类别（原 S1.3 的设定）
RUN_BOTH=0           # 1 = 同时跑 7 类和 8 类，用于敏感性分析

N_SUBSAMPLE=100      # 子采样次数 B
SUBSAMPLE_FRAC=0.80  # 每次子采样的个体比例
REPEATS=10           # panel 评估的重复次数（10 × 5 折 = 50 个测试折）
CV_FOLDS=5           # 折数
COARSE_STEP=10       # 粗筛步长
COARSE_MAX=200        # 粗筛最大规模
REFINE=6             # 转折点前后 ±N 内做单 SNP 分辨率评估
TARGET=1.0           # 四个指标须同时达到的目标值
SEED=20261027
N_JOBS=-1            # 0/-1 = 用满本机核数

## ---------- 工具检测 ----------
PY=$(command -v python3 || true)
if [ -z "$PY" ]; then echo "ERROR: python3 未找到"; exit 1; fi

CAND="$STEP2_OUT/candidates_union.tsv"
if [ ! -f "$CAND" ]; then
  echo "ERROR: 找不到 $CAND（请先完成 step2）"; exit 1
fi

mkdir -p "$OUTDIR"
cd "$OUTDIR"

## ---------- [1/2] 准备基因型矩阵 ----------
echo "[1/2] 准备 1,400 个候选 SNP 的基因型矩阵"
if [ -f "$STEP2/geno.raw" ]; then
  GENO="$STEP2/geno.raw"
  echo "  复用 step2 的基因型矩阵: $GENO"
else
  PLINK=$(command -v plink || true)
  if [ -z "$PLINK" ]; then echo "ERROR: 需要 plink 重建 geno.raw，但未找到"; exit 1; fi
  tail -n +2 "$CAND" | cut -f2 | sort -u > candidates.txt
  "$PLINK" --bfile "$WORKDIR/step1/ld/ld_pruned" --allow-extra-chr \
    --extract candidates.txt --recode A --out geno > /dev/null
  GENO="$OUTDIR/geno.raw"
  echo "  已用 plink 重建: $GENO"
fi

## ---------- [2/2] 稳定性排序 + panel 优化 ----------
run_one () {
  local cl=$1 tag=$2
  echo ""
  echo "========== classes = $cl =========="
  "$PY" "$SCRIPT_DIR/step3_elasticnet.py" \
    --geno "$GENO" \
    --pop "$POP" \
    --candidates "$CAND" \
    --classes "$cl" \
    --n-sub "$N_SUBSAMPLE" --sub-frac "$SUBSAMPLE_FRAC" \
    --repeats "$REPEATS" --folds "$CV_FOLDS" \
    --coarse-step "$COARSE_STEP" --coarse-max "$COARSE_MAX" \
    --refine "$REFINE" --target "$TARGET" \
    --seed "$SEED" --n-jobs "$N_JOBS" \
    --outdir out --tag "$tag"
}

echo "[2/2] multinomial Elastic Net 稳定性排序 + panel 大小优化"
if [ "$RUN_BOTH" = "1" ]; then
  run_one 7 _7class
  run_one 8 _8class
  echo ""
  echo "对比（两者最终 panel 规模）："
  paste <(grep -H final_panel_size out/summary_7class.txt out/summary_8class.txt | sed 's/.*:final_panel_size/final_panel_size/')
else
  run_one "$CLASSES" ""
fi

echo ""
echo "完成。结果在: $OUTDIR/out/"
echo "  ranking.tsv              1,400 个候选的稳定性排序"
echo "  panel_size_metrics.tsv   各 panel 规模的四项指标"
echo "  panel_class_recall.tsv   逐类召回（可直接用于 Fig. 3C）"
echo "  final_panel.tsv          最终 panel 的 SNP 清单"
echo "  summary.txt              超参与最终规模"
