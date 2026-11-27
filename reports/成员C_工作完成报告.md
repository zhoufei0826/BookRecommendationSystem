# 成员 C 工作报告：隐式 ALS、模型融合与提交

更新：2026-11-26。本次按原分工继续完成与 B 的模型比较、简单融合、固定配置复测及全量提交生成。B 的代码来自最新仓库压缩包，导入后未修改；使用其 B-001 参数，没有进行 B 模型的二次调参。入口与复现方式见项目根目录 `README_C.md`。

## 已完成

- 将赛题提供的 `train_dataset.csv`、`test_dataset.csv` 和占位样例 `submission.csv` 复制到 `data/`。原始文件仍保留在上级目录。
- 用成员 A 的 `make_split.py` 按种子 42 生成固定切分：训练 4,717,042 行，验证 1,152,589 行，均覆盖 53,424 位用户。
- 实现 `src/implicit_als.py`：二值隐式反馈 ALS、稀疏矩阵、Numba 共轭梯度更新、Top-10 推荐、已读过滤、原始 ID 还原、与热门度分数融合。
- 使用 A 的 `evaluate.py` 评估；验证目标只参与计算指标，未进入训练或热门度统计。
- 导入并复现 B 的 ItemKNN，使用统一映射进行批量打分，与 ALS 做分数标准化后的简单加权。
- 在 seed 42 比较三个基线与四个融合候选；选定各占 50% 后，在 seed 43 固定权重复测，未重新选权重。
- 用完整训练集重新拟合选定参数，生成 `data/submission_final.csv`。A 的 `check_submission.py` 检查全部通过。

## 第一阶段：ALS 与热门度融合（历史结果）

主切分：seed 42，Top-10，按 micro-F1 选择参数。ALS 使用 `alpha=40`、`regularization=0.1`、`cg_steps=3`、随机种子 42。融合分数为用户内标准化的 ALS 分数与全局标准化的 `log(1+图书交互次数)` 按权重相加。表中权重指热门度占比。

| 因子数 | 迭代数 | 热门度权重 | micro-F1@10 | macro-F1@10 | 图书覆盖率 |
|---:|---:|---:|---:|---:|---:|
| 32 | 10 | 0 | 0.134267 | 0.135291 | 0.3507 |
| 32 | 10 | 0.25 | 0.144898 | 0.145216 | 0.2558 |
| 64 | 12 | 0 | 0.144279 | 0.146296 | 0.4531 |
| 64 | 12 | 0.25 | 0.155693 | 0.156968 | 0.3202 |
| 128 | 12 | 0 | 0.153311 | 0.156754 | 0.5497 |
| **128** | **12** | **0.25** | **0.160504** | **0.163233** | **0.3788** |

完整的 0、0.1、0.25、0.5 四种权重结果已从本地总表提取到 `reports/experiments_C_als.csv`；原始 JSON 保存在 `outputs/c_als_f*_*.json`。GitHub 交接使用 C 的独立实验表，避免覆盖 A 维护的总表。A 报告的同一主切分 BPR-MF micro-F1 为 0.126440；C 的选定方案高 0.034064。该比较是离线指标，不是官方测试分数。

为检查单次切分的偶然性，用相同的按用户切分方法另生成 seed 43。128 因子、12 轮的 micro-F1：ALS 单模型 0.152456，权重 0.25 的融合 0.161917；0.25 在两个切分中均为四个试验权重里最佳。seed 43 用于复测，不替代 A 的主切分。该轮详细结果在 `outputs/c_als_validation_seed43.json`。

## 第二阶段：与 B 比较及融合

固定 B-001：余弦、100 邻居、beta=0.5、无用户加权或候选惩罚。直接运行 B 的脚本得到 micro-F1=0.190614，与其报告 0.190617 接近；融合接口的 ItemKNN 单模型结果同样为 0.190614。以下均使用原切分和 A 的评估器。

| 主切分方案 | micro-F1@10 | macro-F1@10 | 覆盖率 |
|---|---:|---:|---:|
| ItemKNN | 0.190614 | 0.191896 | 0.5265 |
| ALS | 0.153311 | 0.156754 | 0.5497 |
| ALS + 热门度 0.25 | 0.160504 | 0.163233 | 0.3788 |
| ItemKNN 0.9 + ALS 0.1 | 0.192532 | 0.193833 | 0.5283 |
| ItemKNN 0.75 + ALS 0.25 | 0.195843 | 0.197197 | 0.5305 |
| **ItemKNN 0.5 + ALS 0.5** | **0.203490** | **0.204922** | **0.5393** |
| ItemKNN 0.75 + 第一版 ALS/热门度方案 0.25 | 0.194582 | 0.195953 | 0.5223 |

融合前按用户分别标准化模型在完整训练图书目录上的分数；最后统一过滤已读并取前 10。最终配置不再加入额外热门度分量。权重仅在主切分选择，配置保存在 `reports/c_selected_model.json`，同时记录切分 SHA256。

seed 43 固定配置复测：ItemKNN micro-F1=0.188042，融合=0.201052、macro-F1=0.202031。两个切分融合相对 ItemKNN 分别多命中 10,859 和 10,973 条交互，支持保留该融合。此结论只来自两个随机切分，不代表官方测试效果。

本次共 11 条比较/复测记录保存在 `reports/experiments_C_fusion.csv`，供 A 汇总；本次未改动 A/B 的实验表。详细参数、数据/代码哈希、推荐列表和耗时见 `outputs/c_fusion/seed42/` 与 `outputs/c_fusion/seed43/`。主切分共享训练 45.5 秒、整轮 204.3 秒；复测共享训练 44.5 秒、整轮 148.9 秒。各候选共用模型训练，不能把整轮时间视作单个候选的时间。

## 当前提交候选

最终使用完整的 5,869,631 条训练交互重新拟合 ALS 与 ItemKNN，采用上述固定的各占 50% 配置。生成的 `data/submission_fusion.csv` 经 A 的检查器验收后复制为 `data/submission_final.csv`，两者内容一致。第一版候选备份为 `data/submission_before_fusion_20261126.csv`。

当前候选有 534,240 行，53,424 位测试用户各 10 本；列名、用户集合、数量、去重、图书 ID、已读过滤、空值与类型、非占位样例检查全部通过。报告保存在 `reports/C_提交检查报告.txt`。全量训练约 52.3 秒，完整生成流程约 96.0 秒（不含检查器）；配置和文件 SHA256 见 `reports/c_submission_info.json`。这些是本地团队规则下的检查结果，尚无官方测试成绩。

## 复现命令

在项目根目录运行。测试环境为 Python 3.12.4，NumPy 1.26.4、pandas 3.0.1、SciPy 1.13.1、Numba 0.59.1。`implicit` 第三方包不是必需依赖。以下命令中的线程数用于控制本机资源占用，模型随机种子由脚本参数控制。

```powershell
$env:NUMBA_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='4'
# 数据切分已存在时，直接使用；缺失时先运行 python src/make_split.py
python -X utf8 src/ensemble.py validate --config reports/c_selected_model.json --out-dir outputs/c_fusion/selfcheck
python -X utf8 src/ensemble.py submit --config reports/c_selected_model.json --out-dir outputs/c_fusion/full_train --output data/submission_final.csv
python -X utf8 src/check_submission.py data/submission_final.csv
python -X utf8 -m unittest discover -s tests -v
```

复测种子 43：

```powershell
python -X utf8 src/make_extra_split.py --seed 43
python -X utf8 src/ensemble.py validate --config reports/c_selected_model.json --train-path outputs/seed43/train_split.csv --val-path outputs/seed43/val_split.csv --out-dir outputs/c_fusion/selfcheck_seed43
```

## 后续协作事项

- C 的模型实现、比较、融合复测、全量重训和候选检查已完成；A 可将 C 的新实验表并入团队总表。
- 全库 17 项测试通过，包括 ALS 正规方程小样本核验、与 B 推荐接口对照、非连续 ID 还原、已读过滤和热门回退。
- CPU 运行使用 Numba 8 线程、OpenBLAS 4 线程，无需 GPU；评分按 256 位用户分批，峰值内存未测量。
- README 只给出通用 F1 公式，未明确 micro/macro、每人推荐数量及列名大小写。当前 Top-10 等约定仍为团队暂定口径；正式提交前应以实际评测方确认的规则为准。
- `data/` 和 `outputs/` 在 `.gitignore` 中；交接项目时需单独传递原始 CSV、切分文件、验证 JSON 和最终提交 CSV。
