# DILI-PLUS 论文—代码对应关系与长期维护手册

> 文档用途：这是给后续 Codex 对话、项目维护者和论文修改者使用的项目记忆，不是论文正文。
>
> 核心原则：**以当前正式代码实现为技术事实，以可追溯结果文件为数值证据；论文文字不能反向证明代码实现。**
>
> 最后系统审计：2026-08-17
>
> 当前代码 checkpoint：`2bfe6ef`（Checkpoint-01，冻结 Code-00--06 合同）；Code-07 语义清理建立在该提交之上
>
> 论文基线：`000fa11a9b785cc21f24683b02ea2720cdb32d59`；Code-07 本身未修改论文仓库

## 1. 新对话的一分钟上手说明

新对话开始后，先读取本文件，再读取：

1. [`README.md`](../README.md)：正式流水线入口和目录职责。
2. [`REVISION_ROADMAP_PEER_REVIEW.md`](REVISION_ROADMAP_PEER_REVIEW.md)：同行评审意见的专业性校验、AI 幻觉纠偏和分阶段大修任务。
3. [`configs/default.yaml`](../configs/default.yaml)：数据、模型、报告和图片路径。
4. [`pipelines/`](../pipelines)：五个阶段的正式入口。
5. [`src/diliplus/`](../src/diliplus)：唯一应视为当前实现的源代码。
6. `D:\PaperWorks\DILI-PLUS\main.tex`：论文当前正文。

不要把根目录数字编号脚本当成第二套实现。它们是兼容入口；新的分析和修复应进入 `src/diliplus/` 与 `pipelines/`。

### 1.1 两个独立 Git 仓库

| 对象 | 本地路径 | 远程仓库 | 用途 |
|---|---|---|---|
| 代码 | `D:\Code_for_all\aMED_paper_05_DILIPLUS` | `AYHQChang/DILI-PLUS` | 数据构建、模型、评估、解释、制图和本维护文档 |
| 论文 | `D:\PaperWorks\DILI-PLUS` | `AYHQChang/-Double-column-Predicting-DILI-PLUS` | `main.tex`、`references.bib`、投稿文件和论文使用的 PDF 图片 |

二者没有子模块或嵌套关系。代码生成图片后，需要人工复制到论文仓库的 `Figures/`，然后分别提交两个仓库。

### 1.2 证据等级

后续审查每个论文主张时使用以下等级：

| 等级 | 定义 | 可以怎样写进论文 |
|---|---|---|
| A | 当前正式代码直接实现，且有当前数据产物/结果文件支持 | 可陈述，但仍需使用观察性研究的谨慎措辞 |
| B | 当前正式代码直接实现，但结果尚未重跑或成功产物缺失 | 方法可写，数值和效果不能先写死 |
| C | 只有现有结果文件或图片支持，但生成过程存在已知缺陷或版本不明 | 只能作为待复核结果，修复并重跑前不应形成强结论 |
| D | 论文有描述，但当前代码没有实现或实现与描述不同 | 必须改论文，或先改代码并完整重跑 |
| E | 因果、临床有效性或安全性主张超出回顾性代码与数据的识别能力 | 不应主张；只能写为模型敏感性、关联或未来验证假设 |

## 2. 研究动机、现实目标与边界

### 2.1 现实研究动机

本项目面对的是高强度多重用药住院环境中的急性肝损伤代理结局。真实困难包括：

- 极端类别不平衡：当前张量队列包含 51,316 次住院记录，其中 507 个阳性、50,809 个阴性，阳性率约 0.99%。
- 时间轴不对齐：阳性病例以生化阈值首次达到时间为观察终点；若阴性病例一直观察到用药结束或出院，模型可能把观察时长当成标签代理。
- 动态信息异质：用药是离散事件，化验是带数值的连续观测，诊断是静态/半静态表型。
- ICU/高危场景中的肝损伤不是经过因果归因的纯药物性肝损伤；药物暴露、感染、休克和缺血缺氧可能共同参与。
- 模型输出来自观察性 EHR，任何换药、减药或药物归因分析都只能解释模型响应，不能识别真实治疗效应。

### 2.2 代码真正支持的研究目标

当前代码支持的目标应表述为：

> 在基线 ALT/AST 状态为正常或偏低、后续 ALT 或 AST 达到 120 U/L 的多重用药住院队列中，构建时间感知的多模态单任务分类模型，评价其对这一生化急性肝损伤代理结局的判别能力、概率表现、时间遮蔽敏感性和局部模型响应。

### 2.3 当前代码不能直接支持的目标

以下表述超出当前实现或数据识别能力：

- “确诊 DILI”“药物导致了肝损伤”或符合完整 DILIN/RUCAM 因果归因。
- “消除”了 immortal time bias；代码只实施了一种随机截断策略，最多说“试图缓解”。
- DILI–AKI 多任务学习或不确定性加权多任务正则化已经用于最终模型。
- 真实反事实、平均处理效应、绝对风险降低、临床换药建议或治疗安全性。
- 外部验证、前瞻性验证或临床部署有效性。
- 12 层、768 维的 MedBERT 规模，除非代码先按该结构实现并重跑。

## 3. 结局、队列与输入数据的代码事实

### 3.1 多重用药队列

正式实现：[`src/diliplus/data/cohort.py`](../src/diliplus/data/cohort.py)

- 来源：只读 DuckDB 中的 `analysis.feature_medication_seq`、`analysis.feature_medications` 和化验表。
- 纳入：`med_count >= 5`，且首次/末次用药时间非空。
- 肝功能化验池：ALT、AST、总胆红素、直接胆红素、ALP、GGT 等中文项目名匹配。
- 化验提取范围：首次用药后至末次用药后 7 天；这一设计允许捕获迟发事件，但也导致阳性与阴性的可用时间边界不完全对称。
- 输出：`data_cache/01_aligned_dili_labs.parquet` 和 `reports/01_DILI_Cohort_Outcome_Summary.csv`。

证据等级：队列 SQL 为 A；“ICU encounter”需要源数据库场景字段进一步证明，目前代码只直接证明是高强度多重用药住院记录。

### 3.2 生化代理标签

正式实现：[`src/diliplus/data/labels.py`](../src/diliplus/data/labels.py)

- 仅使用名称匹配到 ALT/AST 的化验。
- 每次住院的第一次相关化验以 `abnormal_status` 判断基线，而不是用数值 ULN 统一重算。
- 基线状态必须属于正常/低值集合。
- 后续任一 ALT 或 AST 数值 `>= 120 U/L` 定义正式列 `label_ahi_proxy = 1`；构建器暂时额外写出同值的 `label_dili` 兼容别名。
- 首次达到 120 U/L 的时间定义为 `t_onset`。
- 当前主任务固定 `prediction_gap = 24h`；阳性 `index_time=t_onset`，`prediction_time=t_onset-24h`。
- 阴性在可形成 24h gap 的观察窗内，用 encounter ID 与固定 seed `20260816` 的哈希生成确定性 pseudo-index，再令 `prediction_time=index_time-24h`。
- 没有实施 RUCAM/DILIN 药物归因，没有排除休克、脓毒症、病毒性肝炎等所有替代病因。

Code-07 后，正式 Dataset/训练接口强制要求 `label_ahi_proxy`，并把通用训练键 `label` 明确定义
为同一 AHI-proxy 标签；只有 `label_dili` 的旧 Parquet 会被拒绝。因此论文中应使用
“biochemistry-defined acute hepatic injury proxy”或“AHI proxy”，不是经过临床判定的 DILI。

修复后 24h 数据单独写入 `data_cache/prediction_gap_24h/`，当前为 46,864 encounters、391 AHI-proxy positives。旧 0h/onset-time 数据、checkpoint 和结果不得与该版本混用。

### 3.3 动态药物与化验序列

正式实现：[`src/diliplus/data/sequences.py`](../src/diliplus/data/sequences.py)

- 药物和化验流均只保留严格满足 `event_time < prediction_time` 的事件；同时保存真实 `med_event_times`/`lab_event_times` 和相邻事件时间差。
- 独立 Parquet 审计确认用药越界、化验越界和输入内 ALT/AST `>=120 U/L` 均为 0。
- 输出：`data_cache/prediction_gap_24h/03_dili_dual_stream_tensors.parquet`。

当前本地张量快照：46,864 行；391 阳性、46,473 阴性。

### 3.4 诊断表型和目标致盲

正式实现：[`src/diliplus/data/diagnoses.py`](../src/diliplus/data/diagnoses.py)

- 精确连接路径为 `encounter_id -> visit_number -> business_uu -> inpatient_f`；不再使用患者号取回诊断。
- 仅保留源行 `create_time` 可解析且严格满足 `diagnosis_time < prediction_time` 的诊断。
- 排除 ICD-10 `K71%` 及中文“药物性肝”“毒性肝”“中毒性肝”标签；无时间和边界后诊断不插补。
- 修复前审计发现 30.71% 的同次住院去重诊断位于 prediction time 当时或之后；修复后 216,220 条保留诊断的时间、显式目标和 encounter 对齐违规均为 0。
- `create_time` 只代表数据库记录可用时间代理，不代表疾病发生时间；论文必须保留这一限制，并且后续完成 without-diagnosis 消融。

输出：`data_cache/prediction_gap_24h/03b_diag_tensors.parquet`；详细审计见 `docs/REVISION_ROADMAP_PEER_REVIEW.md` 第 16 节。

### 3.5 患者级分组核验

训练器使用 `encounter_id` 下划线前缀作为 patient group。以下数量是 Code-02 以前 51,316 行
快照的历史核验，不是当前 46,864 行 prediction-gap 队列的性能证据：

- 51,316 次住院记录。
- 51,295 个唯一分组/患者号。
- 21 名患者各有重复住院，共 42 次重复患者住院记录。
- 分组前缀与队列中的 `health_reco` 完全一致。

Code-05 已对当前 46,864 行队列重新建立五折 `StratifiedGroupKFold`，并自动证明每折四方角色的
patient-group overlap 为 0。若将来更换数据编码规则，必须重新验证这一等价关系，不能只相信变量名。

## 4. 模型真实结构

正式实现：[`src/diliplus/models/diliplus_engine.py`](../src/diliplus/models/diliplus_engine.py)

### 4.1 输入和截断长度

数据集实现：[`src/diliplus/data/dataset.py`](../src/diliplus/data/dataset.py)

| 模态 | 输入 | 默认最大长度 |
|---|---|---:|
| 药物 | token、相邻事件时间差、mask | 100 |
| 化验 | 项目 token、连续数值、相邻事件时间差、mask | 20 |
| 诊断 | ICD token、mask | 15 |

药物和化验项目使用同一份动态词表文件，但模型中是两个独立 `nn.Embedding` 层，不是权重共享的同一个嵌入空间。

### 4.2 `TimeAwareMultimodalTransformer` 的实际参数

| 项目 | Code-07 后当前代码事实 | 论文后续必须同步 |
|---|---|---|
| 正式名称 | `TimeAwareMultimodalTransformer`；建议图表简称 TA-MMT | 不再使用会暗示 Med-BERT 权重的 TA-MedBERT |
| 初始化 | from scratch；不加载、继承或依赖 Med-BERT 预训练权重 | 明确随机初始化，不能称预训练模型 |
| 主任务 | 单一 AHI-proxy 二分类头 `ahi_proxy_head` | 不得恢复 DILI–AKI 多任务或 self-calibration 叙述 |
| 隐藏维度 | 默认 128；配置强制 `>=4`、为偶数且能被 head 数整除 | 保持与配置一致，不写 768 |
| 药物流 Transformer | 2 层、4 heads | 不写 12 层 |
| 化验流 Transformer | 2 层、4 heads | 准确写出第二动态流 |
| 静态诊断 | 独立第三模态，经池化后参与门控融合；训练时可逐样本只丢弃诊断表示 | 不把 diagnosis dropout 写成对其他模态的缩放 |
| 时间编码 | 对 `log1p(dt)` 应用可学习频率和相位的 `sin` 分量，加一个线性投影 | 不使用固定 Transformer 正弦/余弦公式 `10000^(2i/d)` |
| 时间变量 | 相邻事件间隔；首事件相对首次用药 | 只称时间间隔表示 |
| 融合 | 三模态拼接、投影、门控和动态残差 | “dual-stream”只能指两个动态流 |
| 输出 | 字典中的单一 `[batch, 2]` AHI-proxy logits + 三个模态表示 | 不写 AKI head、MTL 或 tuple output |

模型应描述为“药物和化验两个动态流，加一个静态诊断模态的时间感知多模态分类器”。如果保留“dual-stream”，必须明确它只指两个动态流，而不是全部输入只有两种。

旧 Python 名 `DILIPlusEngine` 仅为导入兼容别名；Transformer 对照同理以
`MultimodalTransformerBaseline` 为正式名，`MultiModalBaselineMedBERT` 仅为兼容别名。正式
registry、训练 CLI、正式流水线生成的新 artifact 和报告标签不得接受或写入旧名。

### 4.3 时间编码的准确写法

当前代码不是论文中的固定 sinusoidal positional encoding。准确概念是：

1. `dt_norm = log1p(max(dt, 0))`。
2. 周期分量：`sin(dt_norm * omega + phi)`，其中 `omega` 和 `phi` 可学习，`omega` 以 log-uniform 方式初始化。
3. 非周期分量：`Linear(dt_norm)`。
4. 两者相加后再与事件 token embedding 相加。

该编码只能称为可学习的时间间隔表示；不能称为真实药代动力学方程、指数清除模型或剂量反应模型。

## 5. 训练、校准与评价的代码事实

### 5.1 正式校准训练路径

入口：[`pipelines/02_train_models.py`](../pipelines/02_train_models.py)

实现：

- [`src/diliplus/training/deep_trainer_calibrated.py`](../src/diliplus/training/deep_trainer_calibrated.py)
- [`src/diliplus/training/ml_baselines_calibrated.py`](../src/diliplus/training/ml_baselines_calibrated.py)

六种正式实验键：Logistic Regression、XGBoost、`MultiModalTextCNN`、`MultiModalBiLSTM`、
`MultimodalTransformerBaseline`、`TimeAwareMultimodalTransformer`。后四个深度模型名由
`src/diliplus/models/registry.py` 的 formal registry 集中限定；旧 MedBERT 名不属于正式实验键。

Code-05 后，外层使用固定 seed 的 5 折 `StratifiedGroupKFold`。每个 outer-training pool 按 patient group 分成约 70% parameter-training、15% selection、15% calibration；内层从 128 个确定性 `GroupShuffleSplit` 候选中选择同时含两类且规模/患病率最接近母集的方案。selection 只用于早停/模型选择，calibration 只拟合温度，outer test 只生成一次最终 logits。真实 46,864 条队列的全部 index/group overlap 为 0，证据见 `manifests/code05_split_protocol.json`。

### 5.2 当前训练是显式单任务 AHI-proxy 分类

Code-07 已从 active trainer 删除 `UncertaintyMTLLoss`、AKI loss、可学习 `log_vars`、AKI 标签/头和
tuple 输出兼容分支。模型输出由统一 helper 验证为字典中的 `[batch, 2]` AHI-proxy logits，
优化器只接收模型参数。

结论：当前不存在 DILI–AKI homoscedastic uncertainty MTL、自适应任务权重、representational
Nash equilibrium 或由 MTL 导致的 self-calibration。旧论文/旧审查中这些表述仍是历史 D 级
主张；Code-07 的删除是让实现如实收敛为单任务，不是完成了某种 MTL 实验。

### 5.3 Focal Loss 的实现边界

当前正式深度训练使用独立的 `UnweightedFocalLoss`，默认 `gamma=2.0`：

```text
(1 - p_t)^gamma * cross_entropy
```

它不接受 `alpha`，也不应用类别权重。论文只能称为 unweighted focal modulation；不得再写固定
`alpha=0.25`，也不得称为 class-balanced 或 alpha-balanced focal loss。

模型选择使用验证集 AUROC，而不是 AUPRC；论文可说“报告时优先解释 AUPRC”，不能说训练/早停以 AUPRC 为唯一目标。

### 5.4 温度缩放

- 深度模型直接在独立 calibration partition 的两类 logits 上用 L-BFGS 拟合一个严格为正的温度；传统 ML 把同一 calibration partition 的两类概率转换为 log-probability 后使用相同实现。
- calibration partition 不参与参数更新、epoch selection 或 outer-test evaluation。
- outer test 只前向一次；raw 与 calibrated 分别为同一 logits 的 `softmax(logits)` 和 `softmax(logits/T)`。
- 温度、selected epoch、exact split、配置、run ID 和数据指纹与模型共同保存在版本化 artifact 中。
- early-warning、IG 和扰动必须显式指定 run/fold 及 raw/calibrated，且加载时校验数据与 metadata；不再把文件名中的 `calib` 当作已应用温度的证据。

### 5.5 未校准对照路径已停用

Code-05 已删除“每个 epoch 查看 outer test 并按 test AUROC 选权重”的实现。旧 `deep_trainer.py`/`ml_baselines.py` 仅作为兼容入口转发到正式四方协议；formal pipeline 不再允许单独训练 uncalibrated 模型。

新预测文件同折并列保存 `y_prob_raw` 与 `y_prob_calibrated`，两列来自一次 outer-test logits。现有 Figure 3 仍是旧实验产物，必须在 Code-10 用新 run 重做后才能评价温度缩放；代码协议修复不追认旧图。

### 5.6 指标定义

| 指标 | 代码实现 |
|---|---|
| AUROC | `roc_auc_score` |
| AUPRC | `average_precision_score` |
| Brier | `brier_score_loss` |
| Quantile ECE | 10 个等频预测概率分箱 |
| pAUC | FPR 0–0.2 区间积分，再除以 0.2 归一化 |
| NetBenefit AUDC | 阈值 0.01–0.99 的净收益，负值先截为 0 再积分 |

论文必须明确 pAUC 是归一化 pAUC，AUDC 也不是单一临床阈值上的净收益。把 AUDC 称为“临床有效性”属于过度解释；它只是当前数据和模型概率下的统计决策曲线摘要。

## 6. pre-Code-07 历史结果快照与论文数值对应

以下数据源仍是历史 `reports/05_Calibrated_Results_Table.csv`，使用旧数据边界、旧 split/artifact
协议和旧模型命名，不是 Code-05/06/07 后的新协议结果。表中模型名和数值原样保留仅用于
追溯论文旧稿；不能改贴新正式模型名，也不能继续作为修复后性能证据：

| 模型 | AUROC | AUPRC | Quantile ECE | pAUC@FPR≤0.2 | NetBenefit AUDC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9134 ± 0.0232 | 0.2390 ± 0.0520 | 0.0804 ± 0.0036 | 0.7202 ± 0.0476 | 0.000018 ± 0.000010 |
| XGBoost | 0.9251 ± 0.0228 | 0.3192 ± 0.0653 | 0.0043 ± 0.0010 | 0.7663 ± 0.0394 | 0.0012 ± 0.0004 |
| MM-TextCNN | 0.9943 ± 0.0007 | 0.7032 ± 0.0169 | 0.0177 ± 0.0010 | 0.9713 ± 0.0036 | 0.0028 ± 0.0002 |
| MM-BiLSTM | 0.9888 ± 0.0020 | 0.6143 ± 0.0309 | 0.0774 ± 0.0544 | 0.9448 ± 0.0088 | 0.0011 ± 0.0011 |
| MM-BaselineMedBERT | 0.9886 ± 0.0028 | 0.7278 ± 0.0178 | 0.0177 ± 0.0006 | 0.9506 ± 0.0073 | 0.0032 ± 0.0004 |
| TA-MedBERT | 0.9910 ± 0.0018 | 0.7389 ± 0.0402 | 0.0923 ± 0.0416 | 0.9553 ± 0.0089 | 0.0019 ± 0.0011 |

该表与论文 Table `tab:calibrated_performance` 的主体数值一致。需要注意：

- 论文结果正文另写 TA-MedBERT AUPRC 为 `0.7278 ± 0.0705`，与表和 CSV 不一致；`0.7278` 实际接近 BaselineMedBERT 均值，标准差也不对。
- TA-MedBERT 的校准后 ECE 和 AUDC 不是所有模型最优；不能写“optimal calibration”或“highest Net Benefit”。
- 该 pre-Code-07 历史校准表中 TA-MedBERT AUPRC 最高，但与 BaselineMedBERT 的差异是否统计显著尚未做配对检验；新正式模型之间的比较必须由 Code-10 重跑。
- 现有报告只为 AUROC bootstrap 计算置信区间；论文表使用的是折间均值 ± 标准差，不应称作所有指标的 95% CI。

## 7. 论文章节—代码映射

本节同时保留旧稿主张的审计轨迹。凡引用 51,316/507、TA-MedBERT、旧性能 CSV 或旧 Figure
2--4 的行，均应理解为 **pre-Code-07 历史证据**，不得与当前 formal registry 或新 artifact 混用。

| 论文位置 | 论文主张 | 代码/产物 | 审计结论 |
|---|---|---|---|
| Abstract | 51,316、507、0.99% | 旧 `03_dili_dual_stream_tensors.parquet` | C，pre-Code-02/07 历史快照可追溯；当前 24h 队列为 46,864/391，最终数字须正式重跑 |
| Abstract | 右删失消除 immortal time bias | `data/labels.py` | B/D，已实现固定 seed 的确定性 pseudo-index 并完成 5-seed 事件密度审计，但不能宣称“消除”偏倚 |
| Abstract | 双流药物 + 连续生理信号 | `data/sequences.py`、`models/diliplus_engine.py` | A，但还有第三诊断模态 |
| Abstract | AUPRC 0.7389 | `05_Calibrated_Results_Table.csv` | C，pre-Code-07 历史结果可追溯，但修复后须重跑 |
| Abstract | 72h 亚临床检测 | `evaluation/early_warning.py`、`06a...csv` | C，遮蔽逻辑和解释边界需修复后重跑 |
| Abstract | MTL 带来自校准、QECE 0.0923 | `deep_trainer_calibrated.py` | D，MTL 未激活；0.0923 是温度缩放测试结果 |
| Introduction contribution 1 | bounded probabilistic right-censoring | `data/labels.py` | A（实施）/D（“unconfounded”“eliminate”效果） |
| Introduction contribution 2 | continuous harmonic time encoding | `models/diliplus_engine.py` | A（有时间编码）/D（论文公式不符） |
| Introduction contribution 3 | ontology-guided auditing | `explainability/*` | C/E，局部敏感性存在，但实际替换用硬编码表且无因果识别 |
| Method: target blinding | 同次住院、严格预测前诊断及 K71/目标文字屏蔽 | `data/diagnoses.py`、`data/diagnosis_audit.py` | A，真实 Parquet 审计 PASS；`create_time` 作为可用时间代理仍须披露 |
| Method: MTL equation | DILI + AKI | `deep_trainer_calibrated.py` | D，Code-07 后 active code 已明确删除 AKI/MTL 分支 |
| Method: model configuration | 768/12 layers | `models/*.py` | D，实际 128/2 layers |
| Results: model table | 六模型校准指标 | `reports/05_Calibrated_Results_Table.csv` | C，主体数值仅作为 pre-Code-07 历史轨迹可追溯 |
| Results: calibration paradox | 原生 MTL 自校准、温度缩放破坏 TA | raw/calibrated predictions + Figure 3 | D/C，实验不配对、MTL 未激活、ECE 定义混用 |
| Results: 72h | TA AUPRC 0.11277、保留 12.5% | `reports/06a...csv` | C，数值存在，但实验脚本有已知掩码与概率版本问题 |
| Results: IG | Clopidogrel +0.085、Metformin -0.101 | `06b_Target_Patient_Attribution.csv` | C，单病例局部模型归因；不是毒性/保护效应 |
| Results: substitution | Atorvastatin→Pravastatin，预测变化 -4.3 个百分点 | `06c_Counterfactual_Trajectory.csv` | C/E，模型敏感性成立；不是治疗风险降低 |
| Discussion/Conclusion | safe actionable CDSS | 无前瞻或干预验证 | E，应改为潜在审计工具和待验证假设 |

## 8. 表格和图片的生成链路

入口：[`pipelines/05_build_paper_assets.py`](../pipelines/05_build_paper_assets.py)

### 8.1 表格

| 论文对象 | 生成代码/数据 | 当前状态 |
|---|---|---|
| Baseline Table | [`src/diliplus/reporting/table1.py`](../src/diliplus/reporting/table1.py) | Code-08 已成功生成 46,864 encounters / 46,844 patients 的机器可读表、论文格式表、join/schema/baseline audit 和 tracked manifest；论文采用前仍须冻结 legacy label 同时间并列项决策并逐格同步 |
| Architecture Table | 手写在 `main.tex` | Code-07 后须同步正式新名称、from-scratch、128/2 层、单任务和无 alpha 的 loss 合同 |
| Performance Table | `reports/05_Calibrated_Results_Table.csv` | 主要数值可追溯；正文存在一处 AUPRC 错写 |

### 8.2 图片

| 论文图片 | 正式生成模块 | 主要数据源 | 审计状态 |
|---|---|---|---|
| Fig. 1b Data Landscape | `reporting/figures/data_landscape.py` | `03_dili_dual_stream_tensors.parquet` | Code-04 已固定 t-SNE、抽样和散点 jitter seed；仍须基于最终重训数据重新生成并记录输入/输出哈希 |
| Fig. 1c Subgroup Forest | 仅有 `archive/legacy/figures/...` | 版本不明 | `main.tex` 当前未引用；论文仓库存在孤立 PDF，不属于正式资产流水线 |
| Fig. 1d Biomarker Divergence | `reporting/figures/biomarker_divergence.py` | `01...labs` + `02...labels` | Code-04 已固定 seaborn bootstrap seed；72h“divergence”仍只是描述性视觉判断，不是代码检验出的变化点 |
| Fig. 2 Model Comparison | `reporting/figures/model_comparison.py` | 校准逐样本预测 | 当前代码与论文 PDF 渲染像素一致 |
| Fig. 3 Calibration Paradox | `reporting/figures/calibration_impact.py` | 未校准/校准两套预测 | 渲染一致，但实验设计不支持强因果解释；Panel D 对 TA 使用 native、其他模型使用 calibrated，比较不对称 |
| Fig. 4 Early Warning | `evaluation/early_warning.py` → `reporting/figures/early_warning.py` | `06a...csv` | 论文/代码图片仅命名标签不同；结论需修复遮蔽后重跑 |
| Fig. 5 IG Waterfall | `explainability/attribution.py` → `reporting/figures/attribution.py` | `06b...csv` | 渲染一致；正负值仅表示模型输出方向 |
| Fig. 6 Perturbation | `explainability/perturbation.py` → `reporting/figures/perturbation.py` | `06c...csv` | 渲染一致；不是反事实或药效模拟 |

2026-08-16 的 PDF 渲染比较：Fig. 2、3、5、6 和 1c 像素一致；1b 因随机 t-SNE 样本不同而明显不同；1d 因 bootstrap 随机性轻微不同；Fig. 4 只有 “DILI-PLUS”/“TA-MedBERT” 标签差异。不要只靠同名文件判断论文图已同步。

## 9. 早期预警实验的特殊风险

正式实现：[`src/diliplus/evaluation/early_warning.py`](../src/diliplus/evaluation/early_warning.py)

pre-Code-07 历史报告中的 TA-MedBERT：

| Horizon | AUROC | AUPRC |
|---:|---:|---:|
| 0h | 0.99794 | 0.90454 |
| 24h | 0.72460 | 0.18202 |
| 48h | 0.56384 | 0.11811 |
| 72h | 0.54160 | 0.11277 |

72h AUPRC retention = `0.11277 / 0.90454 ≈ 12.5%`，现有 Figure 4 数值由此得到。

修复前必须同时记录：

1. 旧 Figure 4 的 `.pth` 不含温度，历史结果使用 raw softmax；新代码已要求版本化 artifact 和显式概率模式，但尚未正式重跑。
2. TextCNN 不消费更新后的 mask，现有 CSV 中 TextCNN 0/24/48/72h 结果完全相同。正式图已把 TextCNN 排除，但 CSV 仍含无效结果。
3. 修复前只修改 mask，没有同步清零现行键名 `x_med/x_lab/v_lab`。
4. 修复前静态诊断没有按更早 horizon cutoff 重新截断。
5. 0h 输入包含目标定义化验，不能称为无泄露预测基线。
6. 72h AUROC 置信区间跨过 0.5（当前报告约 0.472–0.612），不能写成“statistically robust”而不做正式检验。

Code-09 已改为使用真实 event time 对 medication/laboratory/diagnosis 三模态同步物理截断，
TextCNN 也只消费全部 token 均可见的卷积窗口；当前 24 h 模型数据只允许 effective
24/48/72 h，0/12 h 不能从已截断 artifact 重建。真实 availability audit 已通过，但尚无
Code-10 模型 artifact，因此没有修复后的性能和 Figure 4。在正式重跑前，旧 72 h 数字不能
继续作为当前结果，更不能写“可用于预防性干预”或“已证明 72h 预警有效”。

## 10. 局部解释和扰动分析的真实含义

### 10.1 Figure 5 的病例编号

论文所称 “Patient 44190” 实际是 `DILIPlusDataset` 的零基行索引，不是患者号。CSV 列名 `Patient_ID` 也因此具有误导性。以后应改成 `Case_Index`，论文称“selected case (dataset index 44190)”或使用匿名 case label。

历史行 44190 的本地审计确认其是阳性，并恰好属于旧 Fold 1 测试折。Code-06 后，自动筛选只遍历版本化 artifact 保存的 outer-test indices；手工指定行若不属于该 artifact 的 test fold 会直接失败。新 split 下不得假设 44190 仍属于 Fold 1，必须由新 run 重新选取。

### 10.2 Integrated Gradients

- 使用 Captum `LayerIntegratedGradients`，归因层是 `model.med_embedding`。
- baseline 是全 PAD/0 token。
- 新代码从指定 run/fold artifact 加载模型，并按调用方明确选择 raw logits 或 logits/temperature；历史 CSV 仍来自旧 Fold 1 raw softmax，不能自动升级为 calibrated 结果。
- 同名药物在序列中的归因会聚合。
- 正归因表示该 token 相对于 baseline 推动模型阳性输出；负归因表示降低模型输出。

不能把正归因写成“致病贡献”，也不能把负归因写成“保护作用”。

### 10.3 LOO 与 Track A

- LOO 通过把目标 token 的 `mask_med` 置为 False 测模型输出变化。
- Track A 并不改变真实剂量；它把目标 token embedding 乘以 `alpha`。
- CSV 虽写 `Dose Tapering`，本质是 embedding amplitude attenuation。
- 输出列 `Absolute_Risk_Reduction` 只是同一模型两次预测概率的差，应该重命名为 `Predicted_Probability_Shift`。

### 10.4 Track B 并未使用生成的 Safety_Substitution_Map

[`src/diliplus/explainability/ontology.py`](../src/diliplus/explainability/ontology.py) 会生成 `data_cache/Safety_Substitution_Map.json`，但当前 [`perturbation.py`](../src/diliplus/explainability/perturbation.py) 实际使用模块内手写的 `SUBSTITUTION_MAP_ZH`，没有加载该 JSON。

因此论文当前关于“预定义 ontology/ATC safety map 约束替换”的描述与执行代码不一致。现有 Atorvastatin→Pravastatin 结果只证明：在手写 token 替换规则下，Fold 1 模型对该输入的阳性概率从约 34.09% 变为 29.78%，变化约 -4.31 个百分点。

它不证明：

- Pravastatin 对该患者更安全。
- 真实换药会降低 4.3% 风险。
- 两药临床等效或适应证完全可交换。
- 注意力流形被“证明”保持完整；代码没有计算 latent distance/OOD 指标来验证这一点。

## 11. 高优先级论文—代码不一致清单

### P0：大修前必须处理

1. 删除摘要末尾测试句 `This is TEST.`。
2. 把结局统一为生化定义的 AHI proxy，避免把 `label_dili` 当成经过因果归因的 DILI。
3. 删除或重写 DILI–AKI MTL、自适应 uncertainty weighting 和由 MTL 导致 self-calibration 的所有表述。
4. 把模型规模改为代码实际的 128 维、2 层 Transformer；或先实现论文规模并完整重跑。
5. 把论文时间编码公式改成可学习 `LogUniformTime2Vec + linear` 的实际实现。
6. 校准代码协议已由 Code-05/06 重建为同一 test logits 的配对 raw/calibrated；论文和 Figure 3 仍须在 Code-10 用新 run 重写，不能同时称 0.0923 为 native self-calibration。
7. 修复目标化验进入 0h 输入的问题，并明确 prediction gap/landmark 设计。
8. 修复早期预警遮蔽和温度参数复用后重跑 Figure 4；静态诊断时间界限已于 P0-03 完成。
9. 让 Track B 真正读取经审核的 substitution map，或把论文改成“手工指定 token substitution sensitivity”。
10. 全面删除“safe/actionable treatment guidance”“absolute risk reduction”“pathogenic/protective drug”等超出观察性模型能力的措辞。
11. Code-08 已成功重跑 Table 1；论文尚未逐格同步，且 baseline ALT/AST 同时间并列项与冻结 legacy 标签的差异须在 Code-10 前决策。

### P1：提交前应处理

1. ~~固定 DuckDB pseudo-index、Python、NumPy、PyTorch、DataLoader、t-SNE 和 bootstrap 的随机种子。~~ Code-04 已集中配置并通过两次完整数据重建；跨硬件 GPU 训练仍须在正式 run 中验证。
2. ~~保存数据快照版本、代码提交、配置、运行时间和包版本。~~ Code-00/04 已生成 aggregate-only tracked manifest；当前 dirty worktree 仍须人工复核后提交，manifest 不能代替 commit。
3. ~~不要让结果 CSV 追加旧运行。~~ Code-06 已改为 `reports/runs/<run-id>/`，每模型/折独立覆盖本 run 文件，不读取或追加旧结果。
4. ~~把温度参数与对应 fold/model 一起保存。~~ Code-06 artifact 已保存 temperature、epoch、split、config、run ID 和数据指纹。
5. ~~设立真正独立的 model-selection validation 和 calibration split。~~ Code-05 已建立 training/selection/calibration/test 四方 patient-group 隔离并通过真实队列审计。
6. ~~未校准训练器不得再使用测试折挑选 epoch。~~ 旧路径已停用，兼容入口转发到同一正式协议。
7. 对 `TimeAwareMultimodalTransformer` 与 `MultimodalTransformerBaseline` 的 AUPRC 做患者/折配对的统计比较，不仅比较均值。
8. 修正文中 `0.7278 ± 0.0705` 与性能表不一致。
9. 用正式 run 重做 Figure 4，并统一为 `TimeAwareMultimodalTransformer`/TA-MMT 命名。
10. ~~为诊断特征增加时间和 encounter 边界。~~ 已在 P0-03 改为严格 `diagnosis_time < prediction_time` 并通过真实数据审计。

### P2：维护性和表达

1. 决定是否删除孤立的 Fig. 1c PDF 和 legacy subgroup 脚本，或把亚组分析正式纳入代码与论文。
2. 审核所有参考文献键、年份、DOI 和正文支持关系；本轮没有对外部文献真实性做系统核验。
3. Code-06 已把新 attribution/perturbation 输出改为 `Case_Index`、`Embedding Attenuation`、`Delta_Predicted_Probability_Points`；旧 CSV/图片仍须在 Code-11 正式重跑并替换。
4. 将论文使用图的源代码 commit、输入文件哈希和输出图哈希写入 manifest。
5. ~~把当前 `reports/` 和关键数据产物的可复现摘要以轻量 manifest 形式纳入 Git。~~ Code-00 已新增 `manifests/code00_code04_baseline.json`；最终 figures 仍需在正式重跑时补输出哈希。
6. 把 `pytest` 加入可复现开发/测试依赖。当前 `environment.yml` 未声明 `pytest`，`ML311` 环境无法直接运行完整契约测试；不应通过混用其他 Conda 环境的 `site-packages` 来得出正式测试结论。

## 12. 双向修改规则

### 12.1 修改代码后必须检查论文的情况

| 代码变化 | 必须同步检查的论文内容 |
|---|---|
| 队列纳入/排除标准 | Title、Abstract 样本量/患病率、Methods cohort、Table 1、Fig. 1b、Limitations |
| 标签阈值或基线定义 | 全文 AHI/DILI 定义、onset、目标泄露讨论、所有性能结果 |
| `censor_time`/prediction gap | Methods、Fig. 1b/1d/4、early-warning 文字、Discussion |
| 输入模态或序列长度 | Architecture Methods、配置表、模型示意、ablation 解释 |
| 时间编码公式 | Methods 公式、模型名称、Discussion 中药代边界 |
| 模型层数/hidden size | Architecture Table、Methods、参数量/复杂度表述 |
| loss/任务头 | MTL/Focal Loss/校准叙述、所有结果与消融 |
| CV/split/calibration | Evaluation Protocol、Table 2 caption、Figure 2/3、所有置信区间描述 |
| 指标定义 | Table 2 标题/脚注、Figure 2/3、Results 和 Discussion |
| early-warning mask | Figure 4、72h 主张、Abstract 和 Conclusion |
| IG/LOO/substitution | Figure 5/6 caption、Algorithmic Auditing、因果限制 |
| 图形脚本 | 复制新 PDF 到论文仓库、检查 caption、数值、标签和图中模型名 |

### 12.2 修改论文后必须检查代码/证据的情况

每新增一句方法或结果主张，回答四个问题：

1. 哪个正式 `src/diliplus/` 文件实现它？
2. 哪个输入和输出文件证明它实际跑过？
3. 输出对应哪个代码 commit、配置和数据快照？
4. 这句话是在描述关联、预测、模型敏感性，还是越界暗示因果/临床效用？

若不能回答 1–3，主张最多是待办项，不能写成已完成工作。

## 13. 推荐的标准工作流

### 13.1 纯文字/参考文献修改

1. 修改 `D:\PaperWorks\DILI-PLUS\main.tex` 或 `references.bib`。
2. 本地/Overleaf 编译。
3. 确认只有源文件和必要图片进入 Git；编译产物由 `.gitignore` 排除。
4. 提交并 push 论文仓库。

### 13.2 代码、方法或结果修改

1. 先在代码仓库开工，更新代码、测试和配置。
2. 生成新的数据/结果时必须提供 `--run-id`；训练、评价、解释始终复用同一 run/fold/probability mode，保留旧结果直到核验完成。
3. 运行契约测试和针对该方法的统计核验。
4. 生成表格与图片，记录输入/输出和 commit。
5. 逐项更新本文件中的映射和风险状态。
6. 把最终论文 PDF 图片从代码 `figures/` 复制到论文 `Figures/`。
7. 修改 `main.tex` 中的方法、数值、caption、讨论和局限性。
8. 分别提交代码仓库和论文仓库；在提交信息或本文件变更记录中互相记录 commit。

### 13.3 推送前检查

代码仓库：

```powershell
git status --short
git diff --check
pytest -q
```

论文仓库：

```powershell
git -C D:\PaperWorks\DILI-PLUS status --short --ignored
git -C D:\PaperWorks\DILI-PLUS diff --check
```

禁止使用 `git add -f` 强制加入 LaTeX 编译产物。图片 PDF 是论文源资产，可以正常跟踪；根目录 `main.pdf` 应保持忽略。

## 14. 新对话接手模板

后续可以直接对 Codex 说：

> 请先阅读代码仓库 `docs/PAPER_CODE_ALIGNMENT.md`，以 `src/diliplus/` 为技术真值，并检查代码和论文两个仓库的 Git 状态。我们本次要修改的是【具体章节/具体代码】。开始修改前，请列出会受影响的方法、结果、表格、图片和需要重跑的产物；任何论文主张必须能追踪到正式代码和当前结果文件。

## 15. 当前建议的大修顺序

1. 已完成目标化验/prediction-time、诊断时间边界、随机性、四方 grouped split、版本化 artifact，以及 Code-07 模型/损失语义合同。
2. Code-08 cohort/Table 1 和 Code-09 early-warning/消融合同已完成；先冻结 baseline ALT/AST 同时间并列项/legacy 标签决策，再由 Code-10 执行唯一正式性能 run。
3. 论文后续统一采用 `TimeAwareMultimodalTransformer`/TA-MMT，并只保留当前实现和新实验真正支持的创新点。
4. 用新结果重做 Table 1、Table 2、Figure 1b/1d/2/3/4。
5. 把 Figure 5/6 降级为单病例模型审计，并改掉因果/治疗用语。
6. 最后重写 Abstract、Methods、Results、Discussion 和 Conclusion，使全文只保留代码与证据真正支持的结论。

## 16. 维护记录

| 日期 | 代码 commit | 论文 commit | 记录 |
|---|---|---|---|
| 2026-08-16 | `cb36322...` | `000fa11...` | 首次建立完整论文—代码映射；确认代码为单任务 128 维实现，识别 MTL、架构规模、时间编码、校准、early-warning、Table 1 和扰动因果措辞等主要不一致 |
| 2026-08-16 | `cb36322...` + dirty worktree hash 见 baseline manifest | 未修改 | 完成 Code-00/04：集中随机配置、5-seed pseudo-index 审计、两次完整数据重建、稳定事件/词表 tie-break、当前 grouped split 摘要和 aggregate-only manifest；21/21 tests PASS；未训练模型 |
| 2026-08-16 | `cb36322...` + dirty worktree | 未修改 | 完成 Code-05/06：五折四方 patient-group split、selection/calibration 隔离、同一 test logits 配对 raw/calibrated、版本化 deep/sklearn artifact、run-specific 输出和下游 run/fold/mode/data guard；真实 split 与 artifact smoke PASS；未训练正式模型 |
| 2026-08-17 | `2bfe6ef`（Checkpoint-01）；Code-07 建立于其上 | 未修改 | 冻结 Code-00--06 合同；完成 Code-07：正式模型改为 `TimeAwareMultimodalTransformer`/`MultimodalTransformerBaseline`，旧名仅作 Python 导入兼容；单任务 AHI proxy、unweighted focal (`gamma=2`, no alpha/class weight)、无 AKI/MTL/tuple、逐样本 diagnosis-only dropout、from-scratch；共享配置强制有效 hidden/head 关系；44/44 tests、语义 audit 与 canonical artifact smoke PASS；未训练正式模型，pre-Code-07 结果继续仅作历史证据 |
| 2026-08-17 | `cc7d434` | 未修改 | 完成并冻结 Code-08/09：真实 Table 1 查询 PASS（46,864 encounters、46,844 patients、391 positives；所有 encounter-level join inflation=1），证实全院住院混合场景；24/48/72 h 三模态 strict cutoff 与 6 项最低消融合同 PASS；未训练/未估计性能，baseline lab 并列项标签决策待冻结 |

以后每次完成会改变论文结论的代码修改，都应在此表增加一行。
