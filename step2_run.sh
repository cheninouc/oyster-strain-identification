#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: 本脚本使用了 bash 数组与后台作业，请用 bash 运行："
  echo "  bash $0"
  exit 1
fi

# ============================================================
# step2: 每品系 one-vs-rest Elastic Net 稳定筛选（S1.2）
#        承接 step1 的每品系 top 2000 -> 每品系 top K -> 合并去重
#
# 并行方式：7 个品系各起一个进程（互不依赖），每个进程内部再并行
#           超参数组合与子采样，最后统一合并。
#
# 依赖: plink (1.9, 提取基因型 --recode A)
#       python3 (numpy, pandas, scikit-learn, joblib)
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

## ---------- 配置（按需修改） ----------
WORKDIR="/public1/node3_chcg/zhiwen/new/new_robot"
STEP1="$WORKDIR/step1"
POP="$WORKDIR/vcf/sample_pop"
OUTDIR="$WORKDIR/step2"

STRAINS=(CF JF KF SB SG SO SW)      # 7 个选育系（对应 CS/JS/KS/BS/GS/OS/WS）
                                    # 负类 = 其余 6 品系 + Wild

CAND_TOP_N=2000     # 取 step1 每品系前多少个 SNP 作为候选
TOP_K=200           # 每品系保留数（7 × 200 = 1,400）
N_SUBSAMPLE=100     # 子采样次数 B
SUBSAMPLE_FRAC=0.80 # 每次子采样个体比例
CV_FOLDS=5          # 超参数 CV 折数
SEED=20261027

TOTAL_JOBS=0        # 本步骤可用的总并行度；0 = 自动取 nproc
PER_STRAIN_JOBS=0   # 每个品系进程内部并行度；0 = 自动 = max(1, TOTAL_JOBS/7)
PARALLEL_STRAINS=1  # 1 = 7 品系并行；0 = 串行（与旧版行为一致）

## ---------- 工具检测 ----------
PLINK=$(command -v plink || true)
if [ -z "$PLINK" ]; then echo "ERROR: plink(1.9) 未找到"; exit 1; fi
python3 -c "import sklearn, joblib, pandas, numpy" 2>/dev/null || {
  echo "ERROR: 缺少 python 依赖（scikit-learn / joblib / pandas / numpy）"; exit 1; }

n_strain=${#STRAINS[@]}
if [ "$TOTAL_JOBS" -le 0 ]; then
  TOTAL_JOBS=$( (nproc 2>/dev/null || sysctl -n hw.ncpu) )
fi
if [ "$PER_STRAIN_JOBS" -le 0 ]; then
  PER_STRAIN_JOBS=$(( TOTAL_JOBS / n_strain ))
  [ "$PER_STRAIN_JOBS" -lt 1 ] && PER_STRAIN_JOBS=1
fi

mkdir -p "$OUTDIR/out"
cd "$OUTDIR"

echo "可用并行度: 总计 ${TOTAL_JOBS}，每品系 ${PER_STRAIN_JOBS}，"\
     "品系级并行: $([ "$PARALLEL_STRAINS" -eq 1 ] && echo 开启 || echo 关闭)"

## ---------- 1. 汇总候选 SNP 并集 ----------
echo "[1/4] 汇总 step1 各品系候选 SNP"
tail -q -n +2 "$STEP1"/out/*_top*.tsv | cut -f4 | sort -u > candidates.txt
echo "  候选并集: $(wc -l < candidates.txt) 个 SNP"

## ---------- 2. 提取基因型（additive dosage） ----------
echo "[2/4] 从 LD 剪枝后的 bed 提取基因型 --recode A"
plink --bfile "$STEP1/ld/ld_pruned" --allow-extra-chr \
  --extract candidates.txt --recode A --out geno > /dev/null
n_snp=$(( $(head -1 geno.raw | wc -w) - 6 ))
echo "  geno.raw: $(wc -l < geno.raw) 行（含表头）, ${n_snp} 个 SNP 列"

## ---------- 3. 稳定筛选 ----------
mkdir -p out
common_args=(
  --indir "$STEP1/out"
  --raw geno.raw
  --pop "$POP"
  --strains "${STRAINS[@]}"
  --cand-top-n "$CAND_TOP_N"
  --top-k "$TOP_K"
  --n-subsample "$N_SUBSAMPLE"
  --subsample-frac "$SUBSAMPLE_FRAC"
  --cv-folds "$CV_FOLDS"
  --seed "$SEED"
  --outdir out
)

if [ "$PARALLEL_STRAINS" -eq 1 ]; then
  echo "[3/4] ${n_strain} 个品系并行筛选（每品系 ${PER_STRAIN_JOBS} 核）"
  echo "      各品系日志: out/<品系>.log，可通过 tail -f 观察进度"
  pids=(); names=()
  for S in "${STRAINS[@]}"; do
    python3 "$SCRIPT_DIR/step2_elasticnet.py" "${common_args[@]}" \
      --only-strain "$S" --n-jobs "$PER_STRAIN_JOBS" \
      > "out/${S}.log" 2>&1 &
    pids+=($!); names+=("$S")
  done

  failed=()
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "  ${names[$i]}: 完成"
    else
      echo "  ${names[$i]}: 失败（详见 out/${names[$i]}.log）"
      failed+=("${names[$i]}")
    fi
  done
  if [ "${#failed[@]}" -gt 0 ]; then
    echo ""
    echo "ERROR: 以下品系失败: ${failed[*]}"
    echo "----- tail out/${failed[0]}.log -----"
    tail -n 25 "out/${failed[0]}.log"
    exit 1
  fi
else
  echo "[3/4] 串行筛选（单进程，内部并行 ${PER_STRAIN_JOBS} 核）"
  python3 "$SCRIPT_DIR/step2_elasticnet.py" "${common_args[@]}" \
    --n-jobs "$PER_STRAIN_JOBS"
fi

## ---------- 4. 合并去重 ----------
echo ""
echo "[4/4] 合并各品系结果并去重"
python3 "$SCRIPT_DIR/step2_elasticnet.py" \
  --indir "$STEP1/out" --strains "${STRAINS[@]}" \
  --top-k "$TOP_K" --outdir out --merge-only

echo ""
echo "完成。结果在: $OUTDIR/out/"
echo "  candidates_union.tsv      去重后的候选并集（下游 step3 输入）"
echo "  strain_selected_snps.tsv  7 品系入选明细"
echo "  <strain>_stability.tsv    每品系全部候选的选择频率与排名"
echo "  hyperparameters.tsv       每品系 CV 选定的 C / l1_ratio"