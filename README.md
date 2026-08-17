# DILI-PLUS

DILI-PLUS 是一个面向住院多重用药场景的 DILI 风险建模项目。实现代码采用 `src/` 包布局；数据构建、训练、评估、解释和论文产物分别由五个阶段入口组织。

论文大修、方法变更或图表更新前，必须先阅读 [`docs/PAPER_CODE_ALIGNMENT.md`](docs/PAPER_CODE_ALIGNMENT.md)。该文档以当前正式代码为技术真值，记录了研究目标、论文—代码对应、结果证据、已知不一致和双向同步规则。同行评审意见的专业性校验、幻觉纠偏、P0/P1/P2 任务和分阶段执行计划见 [`docs/REVISION_ROADMAP_PEER_REVIEW.md`](docs/REVISION_ROADMAP_PEER_REVIEW.md)。共享医疗 DuckDB 的项目架构、已核对表接口、encounter 桥、探表方法、时间合同、验证体系和新项目复用模板见 [`docs/DUCKDB_CLINICAL_DATA_ENGINEERING_PLAYBOOK.md`](docs/DUCKDB_CLINICAL_DATA_ENGINEERING_PLAYBOOK.md)；后续使用同一数据库的新项目应先读该手册，但必须重新验证本项目特定的数据假设。

## 推荐运行方式

```powershell
conda activate ML311
python pipelines/01_build_dataset.py
python pipelines/02_train_models.py --run-id dili24h_v1 --include-ablations
python pipelines/03_evaluate_models.py --run-id dili24h_v1 --probability-mode calibrated
python pipelines/04_explain_models.py --run-id dili24h_v1 --probability-mode calibrated --fold 1
python pipelines/05_build_paper_assets.py --stages table_1
```

全部入口均接受 `--config configs/default.yaml`。项目刻意不提供默认“一键跑完”命令，以避免误触发数据库提取、长时间训练或批量覆盖图片。

`run-id` 是正式训练、评价和解释的必填参数。独立“未校准训练”已停用；每个 outer-test
样本只生成一次 logits，再从同一数组得到 raw 和 temperature-scaled 两列概率。模型
artifact 保存于 `checkpoints/runs/<run-id>/<model>/fold_XX.*`，run-specific 预测和指标保存于
`reports/runs/<run-id>/`，不会追加或混入旧结果。

当前默认预测任务使用 `prediction.gap_hours: 24`。修复后的标签和动态序列写入
`data_cache/prediction_gap_24h/`，不会覆盖 0h/onset-time 的历史 Parquet。数据阶段可按
依赖顺序选择执行，例如：

```powershell
python pipelines/01_build_dataset.py --stages labels sequences temporal_audit
python pipelines/01_build_dataset.py --stages diagnoses diagnosis_audit vocabulary
```

该流程强制所有动态事件满足 `event_time < prediction_time`，并在
`reports/p0_02_prediction_gap_24h/` 保存时序泄漏审计。诊断通过同次住院桥连接，仅保留
`diagnosis create_time < prediction_time` 的记录，并在
`reports/p0_03_diagnosis_time_gap_24h/` 保存源审计和最终 Parquet 契约审计。旧模型和旧预测
结果不能与修复后数据混用。Code-05/06 已完成分层划分、校准和 artifact 协议修复；Code-08
已成功重建 cohort/Table 1，Code-09 已完成 strict early-warning 和最低消融执行合同。进入
Code-10 唯一正式 run 前，仍须冻结 Code-08 暴露的 baseline ALT/AST 同时间并列项与 legacy
标签处理决策；当前仍不应引用任何旧性能作为修复后结果。

需要在 VS Code 中看到完整执行过程时，使用 `Terminal -> Run Task`。当前提供诊断构建、
Code-00/04 确定性重建、Code-04 pseudo-index 敏感性、Code-05 真实 split 审计、Code-06
artifact smoke、Code-07 模型/损失语义审计和 recorded unit tests。Code-08/09 可用下列命令
留下相同格式的终端与日志记录：

```powershell
python scripts/run_recorded.py --run-id code08_table1_local -- python pipelines/05_build_paper_assets.py --stages table_1
python scripts/run_recorded.py --run-id code09_contracts_local -- python scripts/audit_code09_contracts.py
```

这些命令的输出会同步保存为 `reports/run_logs/<run-id>.log`，同名 JSON 记录命令、commit、dirty 状态、
process seed、起止时间、退出码和日志哈希。

## 可复现性与基线

全部随机源集中在 `configs/default.yaml` 的 `reproducibility` 段。正式命令应通过 recorded
runner 启动，使 `PYTHONHASHSEED` 和 CUDA workspace 配置在子进程启动前生效：

```powershell
python scripts/run_recorded.py --run-id unit_tests_local --seed 20260816 -- python -m unittest discover -s tests -v
python scripts/run_recorded.py --run-id pseudo_sensitivity_local --seed 20260816 -- python scripts/audit_pseudo_index_sensitivity.py
python scripts/run_recorded.py --run-id deterministic_rebuild_local --seed 20260816 -- python scripts/verify_deterministic_rebuild.py
python scripts/run_recorded.py --run-id baseline_manifest_local --seed 20260816 -- python scripts/build_baseline_manifest.py
python scripts/run_recorded.py --run-id split_audit_local --seed 20260816 -- python scripts/audit_code05_splits.py
python scripts/run_recorded.py --run-id artifact_smoke_local --seed 20260816 -- python scripts/smoke_code06_artifact.py
python scripts/run_recorded.py --run-id model_loss_semantics_local --seed 20260816 -- python -m unittest tests.test_model_and_loss_semantics -v
python scripts/run_recorded.py --run-id code07_semantic_audit_local --seed 20260816 -- python scripts/audit_code07_model_semantics.py
```

可提交的 aggregate-only 基线保存在 `manifests/code00_code04_baseline.json`，包含配置、运行
环境、关键产物哈希、aggregate cohort 摘要、当前 grouped split membership hash，以及 dirty
worktree 的内容指纹；它不包含 patient/encounter ID。`reports/p0_04_reproducibility/` 和
`reports/run_logs/` 是本地详细证据，仍由 `.gitignore` 排除。基线 manifest 可以识别 dirty
本地状态，但不能替代 Git commit；跨机器复现前仍需人工复核并提交代码。

Checkpoint-01 已在代码提交 `2bfe6ef` 冻结 Code-00--06 的数据、split、评价和 artifact
合同。Code-07 在该 checkpoint 之上只清理模型/损失语义和兼容边界，不生成性能结果。

## 评价与模型 artifact 合同

Code-05 使用五折 `StratifiedGroupKFold` outer test。每个 outer-training pool 再按 patient group
划出 15% selection 和 15% calibration：training 只拟合参数，selection 只选择 epoch，
calibration 只拟合一个正温度，test 只执行一次最终推理。aggregate-only split 证据位于
`manifests/code05_split_protocol.json`。

Code-06 artifact 同时保存 model state、四方 split indices/摘要、selected epoch、temperature、
完整配置快照、run ID 和输入数据/词表指纹。early-warning、IG 和 medication-token
perturbation 不再重新计算折分或加载裸 `.pth`；run、fold、数据指纹或 raw/calibrated 模式不
一致时会停止执行。

## 模型与损失语义合同

Code-07 后，正式深度模型名由 [`src/diliplus/models/registry.py`](src/diliplus/models/registry.py)
集中管理：主模型是 `TimeAwareMultimodalTransformer`，Transformer 对照是
`MultimodalTransformerBaseline`；另有 `MultiModalTextCNN` 和 `MultiModalBiLSTM`。历史
Python 类名仅保留为导入兼容别名，不进入 formal registry，也不得写入新的 artifact、报告标签
或论文方法名称。

四个正式深度模型共享同一架构配置；配置加载时强制 `hidden_size >= 4`、为偶数且能被
`num_heads` 整除，避免某个已登记模型在极端配置下产生无效层宽。

所有正式深度模型均从随机初始化开始训练，不加载或继承 Med-BERT 权重。任务是单一
AHI-proxy 二分类；唯一预测输出是字典中的两类 logits，主模型可同时暴露三个模态表示供
受限的解释/敏感性代码复用；不再存在 AKI head、uncertainty MTL、可学习任务权重或 tuple
输出分支。正式 Dataset 强制要求
`label_ahi_proxy`，拒绝只有 `label_dili` 的旧产物；数据构建器暂时额外写出 `label_dili` 作为
下游兼容别名，但正式训练只使用 AHI-proxy 标签语义。

正式损失为 [`UnweightedFocalLoss`](src/diliplus/training/losses.py)：默认 `gamma=2`，计算
`(1-p_t)^gamma * cross_entropy`，不接受 `alpha`，也不使用类别权重。主模型的诊断模态
dropout 按样本只清零诊断表示；不会连带缩放药物或化验表示。Code-07 没有正式训练，所有
pre-Code-07 checkpoint、预测、表格和图片仍是历史证据，不能因名称清理而升级为当前结果。
可提交的 aggregate-only 语义证据位于 `manifests/code07_model_loss_contract.json`；审计只用
固定合成输入在 CPU 验证结构，不读取患者数据、不训练，也不计算性能指标。

## Cohort、Table 1 与 early-warning/消融合同

Code-08 以修复后的 `03_dili_dual_stream_tensors.parquet` 为唯一 cohort anchor；诊断、住院维表、
patient profile 和 aligned baseline labs 逐阶段连接。每一步验证 schema、one-row-per-encounter、
匹配数和 row inflation，0 匹配或 N 改变会直接失败。真实结果为 46,864 encounters、46,844
unique patients、391 positives；完整 aggregate 证据见 `manifests/code08_cohort_table1.json`，
通用排错流程见 DuckDB playbook 第 9.7 节。

Code-09 使用每个事件到 prediction time 的真实时距，同时截断并清零 medication、laboratory
和 diagnosis 输入。由于正式数据已经在 index time 前 24 h 截断，当前只允许 effective
24/48/72 h，不能从现有 artifact 伪造 0/12 h 输入。最低消融矩阵包含 static diagnosis-only、
medication-only、laboratory-only、full without time、full without diagnosis 和 full primary；
只有显式传入 `--include-ablations` 才会在正式 run 中训练五个 ablation-only 模型。Code-09
tracked manifests 只证明输入和结构合同，不含训练或性能结果。

## 目录职责

```text
configs/          集中路径与训练默认值
pipelines/        五个可执行阶段入口
src/diliplus/     唯一的当前实现
tests/            无数据库、无训练、无绘图的契约测试
manifests/        可跟踪的 aggregate-only 基线与正式运行摘要
archive/legacy/   不进入新流水线的历史代码
```

根目录中的数字编号脚本是兼容入口，用来保证旧命令、实验笔记和批处理不失效。它们不包含第二份算法，也不会重复执行计算；新工作应直接使用 `pipelines/`。

## 产物兼容合同

默认配置仍将产物写入原目录：Parquet 到 `data_cache/`，词表到 `vocab/`，权重到 `checkpoints/`，预测与研究表格到 `reports/`，PNG/PDF 到 `figures/`，药物映射 CSV 留在项目根目录。相对路径始终以项目根目录解析，与启动时的当前工作目录无关。

源医疗 DuckDB 只能经 `diliplus.database.connect_source_database` 连接，并固定使用 `read_only=True`；配置文件不能将其改为写模式。
