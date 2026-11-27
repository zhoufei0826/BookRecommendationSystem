# 成员 C：ALS、模型融合与提交候选

本说明对应《项目分工与分阶段实施方案》中成员 C 的职责：实现隐式 ALS、进行有限调参、与 A/B 的模型比较、验证简单融合、全量重训并生成提交 CSV。

本地已接入最新代码包中的 B 成员实现。C 的融合直接调用 B 的 `compute_similarity()` 和 `truncate_neighbors()`，沿用 B-001 的参数；A 的切分、评估器与提交检查器继续复用。原始赛题说明见 [README.md](README.md)。

## 当前结果

以下均为团队暂定的 **micro-F1@10 离线指标**。主切分为 A 的 seed 42；seed 43 只复测主切分选定的融合配置，没有重新选择权重。

| 模型 | 主切分 seed 42 | 复测 seed 43 |
|---|---:|---:|
| A 的 BPR-MF（引用 A 报告） | 0.126440 | 本次未复测 |
| C 的 ALS 单模型 | 0.153311 | 0.152456 |
| C 第一版 ALS + 热门度 | 0.160504 | 0.161917 |
| B 的 ItemKNN（本地复现） | 0.190614 | 0.188042 |
| **ItemKNN 与 ALS 各占 50%** | **0.203490** | **0.201052** |

最终采用 `score = 0.5 × z(ItemKNN分数) + 0.5 × z(ALS分数)`。`z` 表示对同一用户在训练图书目录上的分数做均值、标准差归一化；融合后过滤训练历史并选 Top-10。主切分相对 ItemKNN 多命中 10,859 条验证交互，F1 提高 0.012875。融合权重只是本次有限比较中的最佳配置。

- ALS：128 因子、12 轮、alpha=40、regularization=0.1、cg_steps=3、模型种子 42。
- ItemKNN：B-001，余弦相似度、100 个邻居、历史热度降权 beta=0.5，不加相似度收缩或候选热度惩罚。
- 固定配置保存在 [reports/c_selected_model.json](reports/c_selected_model.json)。
- 早期 ALS 实验见 [reports/experiments_C_als.csv](reports/experiments_C_als.csv)，与 B 的模型比较及融合见 [reports/experiments_C_fusion.csv](reports/experiments_C_fusion.csv)；说明见 [成员 C 工作报告](reports/成员C_工作完成报告.md)。

## 复现

在包含 `src/` 的项目根目录运行。使用已安装 NumPy、pandas、SciPy、Numba 的 Python。测试环境：Python 3.12.4、NumPy 1.26.4、pandas 3.0.1、SciPy 1.13.1、Numba 0.59.1；无需 GPU 或第三方 `implicit` 包。

先把赛题原始 CSV 放在 `data/`。如果固定切分尚未生成，运行 `python src/make_split.py`；已有团队固定切分时直接复用。

```powershell
$env:NUMBA_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '4'

# 按已选配置复跑主切分，结果写到 selfcheck，保留原实验记录
python -X utf8 src/ensemble.py validate --config reports/c_selected_model.json --out-dir outputs/c_fusion/selfcheck

# 独立切分复测；已有 outputs/seed43 时可跳过生成步骤
python -X utf8 src/make_extra_split.py --seed 43
python -X utf8 src/ensemble.py validate --config reports/c_selected_model.json --train-path outputs/seed43/train_split.csv --val-path outputs/seed43/val_split.csv --out-dir outputs/c_fusion/selfcheck_seed43

# 按固定配置全量重训、输出提交候选，再运行 A 的检查器
python -X utf8 src/ensemble.py submit --config reports/c_selected_model.json --out-dir outputs/c_fusion/full_train --output data/submission_final.csv
python -X utf8 src/check_submission.py data/submission_final.csv

# 运行 A/B/C 的小样本测试
python -X utf8 -m unittest discover -s tests -v
```

如需完整复跑本次有限权重比较，可运行下列命令。它会重新写入选定配置文件，宜在正式选型时使用：

```powershell
python -X utf8 src/ensemble.py validate --out-dir outputs/c_fusion/seed42
```

## C 的交付文件

| 文件 | 用途 |
|---|---|
| `src/implicit_als.py` | ALS 和第一版热门度融合 |
| `src/make_extra_split.py` | 使用 A 的切分函数生成独立复测切分 |
| `src/ensemble.py` | 与 B 的模型比较、融合、固定配置复测、完整训练及推荐生成 |
| `tests/test_implicit_als.py`、`tests/test_ensemble.py` | ALS 数值检查、融合 ID 对齐、已读过滤和热门回退检查 |
| `reports/c_selected_model.json` | 在主切分选定的最终配置 |
| `reports/experiments_C_als.csv` | 早期 ALS 与热门度实验记录，单独保存，交由 A 汇总 |
| `reports/experiments_C_fusion.csv` | C 本次比较和复测记录，交由 A 汇总 |
| `reports/c_submission_info.json` | 全量训练配置、耗时、数据/代码/结果哈希 |
| `reports/C_提交检查报告.txt` | A 的检查器对本次候选的检查结果 |
| `data/submission_final.csv` | 当前选定的提交候选，534,240 行 |

本次全量训练约 52.3 秒，完整训练至候选文件写出约 96.0 秒，不含随后运行的检查器；耗时与机器和负载有关。评分按 256 位用户分批进行，不保存完整用户×图书稠密预测矩阵。峰值内存未测量。17 项单元测试与提交检查均通过。

## 当前限制与交接

官方 micro/macro 口径、每人推荐数量和列名大小写尚未获得明确确认。本项目当前沿用团队的 Top-10、`user_id,item_id`、过滤已读约定；本地检查通过不表示平台已验收，离线分数不代表官方测试分数。

`data/` 与 `outputs/` 被 `.gitignore` 排除，仓库中提交代码、配置、报告即可；原始数据与提交 CSV 单独交接。本轮原始结果位于 `outputs/c_fusion/seed42/`、`seed43/`、`full_train/`。此前 ALS+热门度候选保存在 `data/submission_before_fusion_20261126.csv`，当前 `submission_final.csv` 与 `submission_fusion.csv` 内容一致。
