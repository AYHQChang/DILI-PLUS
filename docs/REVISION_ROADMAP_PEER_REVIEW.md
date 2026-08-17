# DILI-PLUS 同行评审校验与论文大修路线图

> 文档用途：把 AI 生成的同行评审意见转换为可核验、可执行、可追踪的代码—实验—论文联合修改计划，供后续 Codex 对话和项目维护者直接接手。
>
> 总原则：**代码和可追溯结果是技术事实；审稿报告负责提出问题，但不能替代代码证据、临床定义或统计验证。**
>
> 审查日期：2026-08-16
> 代码基线：`cb36322ffb4f55a79203af9f630fd48c4af67ebb`
> 论文基线：`000fa11a9b785cc21f24683b02ea2720cdb32d59`

## 1. 本文档审查了什么

本轮综合了以下材料：

1. ChatGPT 对话“文章审查与修改建议Chang大修意见”（conversation ID：`6a771afd-8818-83e8-a4ad-266d6f0cb140`）中可读取的 DILI 审稿意见。
2. 题为“学术同行评审报告：TA-MedBERT用于重症多重用药相关肝损伤预测”的报告中，在本项目 [`DILI PLUS改进方案.md`](../DILI%20PLUS改进方案.md) 留存的六项核心批评和重构建议。
3. 当前论文 `D:\PaperWorks\DILI-PLUS\main.tex`。
4. 当前正式实现 `src/diliplus/`、正式流水线 `pipelines/`、配置、报告和图片产物。
5. [`PAPER_CODE_ALIGNMENT.md`](PAPER_CODE_ALIGNMENT.md) 中已经完成的论文—代码系统审计。

重要限制：被引用对话的最早一轮只返回了用户请求和“附件未包含”的占位符，没有返回那份报告的完整附件正文。因此，本文件只审查当前能够恢复的报告主张，不假装完成了对缺失原文的逐句核对。如果以后重新提供完整报告，应在本文件追加“增量审查”，不应推翻已经由代码确认的事实。

## 2. 总体判断

### 2.1 报告是否专业

**总体上专业，而且抓住了数个真正会影响论文有效性的核心问题。** 最有价值的部分是：

- 区分“达到肝酶阈值的生化性肝损伤”和经过竞争病因排除、药物因果归因的 DILI。
- 识别目标化验进入输入所造成的 target leakage 风险。
- 要求校准参数拟合和最终性能评价相互隔离。
- 区分一般时间间隔编码和真实药代动力学模型。
- 区分模型输入扰动与治疗反事实/因果效应。
- 质疑随机右删失是否改变阴性患者的观察信息结构。

这些问题与 AASLD DILI 指南、TRIPOD+AI、PROBAST+AI 以及预测模型校准的基本原则一致，足以支持“必须大修”的总判断。

### 2.2 是否存在 AI 幻觉或过度推断

**存在，而且不止是措辞问题。** 报告中有些内容把“风险”写成“已经证实的错误”，把旧版本结果写成当前事实，或凭经验猜测修改后的数值。典型例子包括：

- 从脓毒症/休克比例较高直接断言标签“实际上就是 shock liver”。代码只能证明结局没有排除该竞争病因，不能证明每个阳性病例的真实病因。
- 断言校准集与测试集重合。当前正式 calibrated pipeline 的外层测试折是独立的；真正的问题是同一内部验证集同时承担早停/模型选择和温度拟合，以及另一条 uncalibrated pipeline 使用外层测试折挑选 epoch。
- 预测“修复后 ECE 大约会是 0.02”或“绝不可能是 0.0008”。任何尚未运行的结果数值都属于无证据猜测。
- 把一般谐波/可学习时间编码描述为“违背药代动力学”。它不等于药代模型，但可以作为一般时间表示；错误在于论文赋予了超出实现的药代解释，而不是使用该编码本身违反科研规律。
- 把均匀右删失必然称为“拓扑扭曲”“超级幸存者偏差”，并直接跳到 target trial emulation。前者是需要敏感性分析的设计风险；后者只有在研究目标升级为治疗因果效应时才是必要方向，不是当前预测论文的自动修复方案。
- 把项目称为 ICU/critical care，并引用旧稿的密度、患病率或性能数值。当前代码没有直接证明所有记录均为 ICU；旧数值不能自动转移到当前快照。

### 2.3 最终采用方式

这份报告应被视为一份**高价值的问题生成器**，而不是权威裁决。处理规则如下：

| 报告内容类型 | 处理方式 |
|---|---|
| 能由代码、数据产物或权威指南直接支持 | 纳入 P0/P1 修改任务 |
| 方向正确但论证或措辞过强 | 保留问题，改写原因和修复方案 |
| 使用旧稿数值、旧架构或当前无法复现的事实 | 标记“版本过时/待核验”，不得写入结果 |
| 预测未来实验数值或断言真实病因 | 视为 AI 推测，不采纳 |
| 建议扩展为因果推断、ODE、外部验证等新研究 | 作为可选增强项，不阻塞当前预测论文的最小可信版本 |

结论：**报告的“大修”判断成立，但具体修改必须按当前代码重新定义，不能照抄报告原方案。**

## 3. 报告核心意见逐条校验

| ID | 报告主张 | 审查结论 | 当前证据 | 校准后的修改方向 |
|---|---|---|---|---|
| R1 | 当前结局不是严格 DILI | **成立，是最高优先级问题** | [`labels.py`](../src/diliplus/data/labels.py) 仅以基线状态正常/低值、随后 ALT 或 AST `>=120 U/L` 定义阳性；没有 RUCAM/DILIN 归因或完整竞争病因排除 | 论文统一改称 biochemistry-defined AHI proxy；除非另行开展临床判定，否则不得声称确诊 DILI 或 polypharmacy-induced injury |
| R2 | 阳性病例实际上是 shock liver | **不成立为事实，只能作为替代解释** | 数据中可能存在休克/脓毒症，但当前标签不能判定病因 | 写成“不能排除缺血缺氧、感染、心衰等竞争机制”，不能反向断言已经识别 shock liver |
| R3 | 目标化验导致 target leakage | **成立** | [`sequences.py`](../src/diliplus/data/sequences.py) 使用 `lab_time <= censor_time`；阳性 `censor_time=t_onset`，所以定义结局的阈值化验进入 0h 输入 | 引入明确 prediction time/gap；所有模态只允许使用严格早于预测时点的信息；0h 只能标为并发检测/诊断性分析，不能作为前瞻预测主结果 |
| R4 | AST 在 -72h 已越过 ULN，所以 72h 全是泄露 | **当前证据不足** | Figure 1d 的聚合轨迹不能证明每个样本、每折输入均泄露；当前 72h AUROC 约 0.542 且区间跨 0.5，反而不支持“稳健预测” | 修复遮蔽与时间边界后按固定 horizon 重跑；报告每个 horizon 的 N、阳性数、prevalence 和 CI |
| R5 | calibration set 与 test set 重合 | **对当前 calibrated pipeline 不成立** | 外层 5-fold `GroupKFold` 测试折独立；内部 15% 验证集拟合温度 | 保留外层独立测试；把 model-selection validation 与 calibration split 再分开，或诚实说明共用；修复 uncalibrated trainer 使用测试折选 epoch 的真正泄露 |
| R6 | 建议固定 70/10/20 划分 | **可行但不是唯一或最优方案** | 阳性仅 507 例，单次固定划分会降低估计稳定性 | 优先采用患者级 nested/repeated grouped CV；每个外层训练部分再分 training、selection validation、calibration，外层 test 只评一次 |
| R7 | Time2Vec 违背药代动力学 | **方向正确但表述错误** | 当前实现是 `log1p(dt)` 后的可学习 `sin(dt*omega+phi)` 加线性投影；它没有 dose、route、half-life 或 concentration | 把它定义为 learned interval representation；删除药代/清除/累积机制主张；是否增加指数衰减或 ODE 是可选新实验，不是修复当前错误的必要条件 |
| R8 | 论文时间编码公式需要修改 | **成立，而且报告还没抓到完整差异** | 论文写固定 `10000^(2i/d)` 正弦/余弦；代码是 learnable LogUniformTime2Vec + linear | 以代码公式重写 Methods，或先改代码并完整重跑；不得只改一个术语 |
| R9 | Track A 的 embedding amplitude 不能当剂量 | **成立** | [`perturbation.py`](../src/diliplus/explainability/perturbation.py) 直接缩放 embedding，未编码 dose | 统一命名为 embedding attenuation/suppression sensitivity；删除 tapering、dose reduction、withdrawal 等临床解释 |
| R10 | Track B 不是真实 counterfactual/ARR | **成立** | 仅替换 token 后比较模型概率，没有处理适应证混杂、时间变化混杂、可交换性或 positivity | 报告 `predicted probability shift under token substitution`；不得叫 ARR/ATE/安全替药建议 |
| R11 | ontology 能保证替换“安全、在流形内” | **不成立** | 当前 perturbation 甚至使用代码内硬编码映射，没有读取生成的 `Safety_Substitution_Map.json`；也没有 latent/OOD 距离检验 | 要么读取并人工审核映射并仅做模型敏感性审计，要么删除 ontology/safety/manifold-preserving 主张 |
| R12 | 随机右删失必然消除了 immortal time bias | **不成立；只能说尝试缓解** | 阴性在用药起止之间用未设种子的 DuckDB `RANDOM()` 均匀抽取时点 | 固定随机种子，比较多种 pseudo-index 方案；检查观察时长分布、阳性率和性能敏感性；论文使用“mitigate”，不用 eliminate/guarantee/unconfounded |
| R13 | 应直接上 target trial emulation | **不作为当前论文必需项** | 当前问题是预测/模型审计，不识别治疗效果 | 只有未来要回答“换药会不会降低真实风险”时才设计 target trial/因果研究；当前先守住预测边界 |
| R14 | patient-level split 可能缺失 | **当前快照已核验为成立** | 51,316 encounters、51,295 patient groups；下划线前缀与 `health_reco` 一致，重复患者未跨折 | 保留并加入自动契约测试；更换数据后必须重新核验 |
| R15 | 模型可称为 ICU/high-acuity 模型 | **不成立** | Code-08 科室审计确认是全院住院混合 cohort：重症/监护名称筛查 447/46,864，其余/未知 46,417 | 使用 inpatient/hospitalised polypharmacy cohort；不得写 ICU-only |

## 4. 报告没有充分指出、但当前更严重的问题

这些问题必须与报告意见一起进入大修，不能因为报告未提及就忽略。

### 4.1 论文中的 MTL/self-calibration 在代码中没有发生

- `UncertaintyMTLLoss` 虽然存在，但当前模型只有一个 AHI/DILI 二分类头。
- 实际训练直接调用 `criterion.loss_fn_dili(...)`，没有 AKI 标签、AKI head，也没有让 `log_vars` 进入计算图。
- 论文中的 DILI–AKI homoscedastic uncertainty MTL、representational Nash equilibrium、organic self-calibration 均没有实现证据。

默认修复：**删除论文中的虚构 MTL 叙述，而不是为了保住文字临时添加一个缺乏研究问题支撑的 AKI 任务。** 若未来确实要做 MTL，应作为新实验从标签、模型、损失、消融和结果全部重新设计。

### 4.2 架构规模和公式不一致

- 当前 hidden size 为 128，不是 768。
- 药物和化验 Transformer 均为 2 层、4 heads，不是 12 层。
- 模型包含药物动态流、化验动态流和静态诊断第三模态；“dual-stream”只适合描述两个动态流。
- 药物和化验虽使用同一词表文件，但模型使用两个独立 embedding 层，不是共享权重空间。
- 初审时实现没有 Med-BERT 式预训练或继承公开 Med-BERT 权重，“TA-MedBERT”会使读者误以为是预训练模型；Code-07 已采用中性的正式名称并把旧名降为 Python 导入兼容别名。

### 4.3 诊断特征存在时间泄露风险

- 原实现只排除 K71/药物性肝损伤显式代码，并按患者号连接，未实施 encounter/time 契约。
- P0-03 源审计证实：严格同次住院的 312,700 条去重诊断中，96,016 条位于 prediction time 当时或之后；应用旧 K71/文字过滤器后仍有 95,857 条非目标晚诊断（占非目标诊断 30.715817%）。391 个阳性 encounter 中 316 个含会穿过旧过滤器的 prediction-time-or-later 诊断，311 个含 onset-time-or-later 诊断。旧诊断模态和依赖它的旧性能结果因此失效。
- 2026-08-16 已改为精确住院桥接、可解析 `create_time < prediction_time` 和显式目标诊断排除；修复后 216,220 条输入诊断的越界与目标诊断违规均为 0。完整证据见第 16 节。

### 4.4 校准结果与 Figure 3 的逻辑互相冲突

- 历史正式表是温度缩放后的外层测试结果；它尚未按修复后协议重跑。
- 历史 Figure 3 的 raw 与 calibrated 预测来自两套独立训练流水线，不是同一模型同一测试 logits 的缩放前后配对。
- 修复前 uncalibrated trainer 在每个 epoch 查看外层 test AUROC，并用它选择最佳 epoch；Code-05 已停用该路径。
- 修复前温度没有保存在 checkpoint，下游加载 `best_calib_*.pth` 后仍使用 raw softmax；Code-06 已改为版本化 artifact 和显式 probability mode。
- 论文一处说 temperature scaling 有效，另一处又说它破坏 TA、证明 native self-calibration，无法由现有实验支持。

### 4.5 “最佳校准/最高临床净获益”与结果表不符

当前五折结果中：

- TA 的 AUPRC `0.7389 ± 0.0402` 是当前表内最高，但差异是否显著尚未做患者/折配对检验。
- TA 的 QECE `0.0923 ± 0.0416` 并非最优；XGBoost 约为 `0.0043 ± 0.0010`。
- TA 的 AUDC `0.0019 ± 0.0011` 也并非最高；BaselineMedBERT 约为 `0.0032 ± 0.0004`。
- 论文正文另有 TA AUPRC `0.7278 ± 0.0705` 的错配，其中 `0.7278` 实际接近 BaselineMedBERT 的均值。

因此，“optimal organic self-calibration”“highest Net Benefit”“superior overall”均需删除或由新实验重新证明。

### 4.6 Code-09 修复前的 Early-warning 流程不能支持 72h 稳健性主张

- 当前 TA 结果约为：0h AUROC/AUPRC `0.998/0.905`，24h `0.725/0.182`，48h `0.564/0.118`，72h `0.542/0.112`。
- 72h AUROC 置信区间跨越 0.5，不能称 statistically robust。
- 修复前 TextCNN 不消费时间 mask，因而不同 horizon 结果不变；这说明旧通用 early-warning 评估接口没有对所有模型实现同一语义。
- 修复前静态诊断在 horizon mask 中从未被时间限制。
- 修复前 early-warning 使用 raw softmax，没有应用各折温度。

Code-09 已在第 20 节完成输入/cutoff 和 artifact 语义修复，但没有正式模型 artifact，因而尚未
生成新性能或 Figure 4；上列历史数字仍不可引用为修复后结果。

### 4.7 Code-08 前 Table 1 和部分图表缺少可复现产物

- 旧 `reports/Table_01_Baseline_Characteristics_DILIPLUS.txt` 只记录 SQL 中断。Code-08 已在第
  20 节生成新的机器可读 Table 1、分阶段 join audit 和 tracked manifest；论文仍须在标签合同
  决策和 Code-10 前后逐格同步，旧失败文件不得继续作为证据。
- Figure 1b 采样/jitter 未完全固定；Figure 1d bootstrap 也有轻微随机差异。
- Figure 1c 文件存在但正文未引用，属于 legacy 资产。
- 报告 CSV 会追加旧运行，可能混合不同版本结果。

### 4.8 解释性结果仍有过度含义

- “Patient 44190”是数据行索引，不是患者标识。
- IG 以全 PAD 作为 drug baseline，方向只代表模型输出变化，不代表 pathogenic/protective effect。
- 解释脚本先扫描全数据选样本，未把“只从当前外层测试折选病例”写成硬约束。
- 单一患者案例不能证明 cohort-level 稳定解释。
- 未测量 latent distance/OOD score，不能声称 substitution preserved the manifold。

## 5. 本轮科学定位：代码允许我们写成什么论文

### 5.1 推荐的最小可信研究目标

> 在基线 ALT/AST 状态正常或偏低、后续 ALT 或 AST 达到 120 U/L 的多重用药住院队列中，构建包含药物事件、实验室观测和既往/同期诊断信息的时间感知单任务模型，评价其对生化性急性肝损伤代理结局的判别、概率校准、预测时距衰减以及模型输入敏感性。

### 5.2 当前允许的贡献

1. 异步药物事件和实验室观测的时间感知联合表示。
2. 极端不平衡住院队列中，对 discrimination、calibration 和 decision-analytic metrics 的并行评价。
3. 不同 prediction gap 下性能衰减的诚实边界分析。
4. IG、LOO 和 token substitution 作为模型行为审计，而不是药物效应估计。

### 5.3 当前禁止或必须降级的主张

- 确诊 DILI、polypharmacy-induced hepatotoxicity、真实药物病因。
- “消除” immortal time bias、完全 unconfounded evaluation。
- DILI–AKI MTL、自适应 uncertainty weighting、self-calibration。
- 真实药代动力学、剂量递减、药物撤停模拟。
- ARR、ATE、safe substitution、therapeutically equivalent recommendation。
- external/clinical validation、patient benefit、actionable CDSS、prophylactic intervention。
- 所有 ICU/critical-care 特定结论，直到数据来源被明确证明。

## 6. 大修决策门

这些决策必须在相关代码开始修改前固定，避免不同阶段反复改研究问题。

| 决策门 | 推荐选择 | 不推荐的捷径 | 影响范围 |
|---|---|---|---|
| D1 结局名称 | Biochemistry-defined AHI proxy | 继续把变量名 `label_dili` 当临床 DILI | Title、Abstract、Methods、所有表图标签、Discussion |
| D2 研究任务 | 单任务 AHI 风险/并发检测；以有 prediction gap 的任务为主结果 | 为迎合现有文字临时拼接 AKI MTL | 标签、模型、loss、训练、校准、消融 |
| D3 主预测时点 | **已固定 24h 为首个主 prediction gap**；0/12/48/72h 保留为可行性/敏感性时距，不按性能事后挑选 | 看完结果后挑表现最好 horizon | 数据构建、early warning、摘要数值 |
| D4 校准设计 | 外层 grouped test；内部 selection 与 calibration 分离 | 同一 test 反复看、两套训练结果伪装成前后校准 | trainer、checkpoints、Figure 3、Methods |
| D5 模型名称 | TA-MedBERT-inspired 或更中性的 TA dual-stream/multimodal model | 暗示使用 Med-BERT 预训练权重 | Title、Methods、baseline 命名 |
| D6 场景名称 | 先核验科室/场景；核验前使用 hospitalised/inpatient | 继续无证据写 51,316 ICU encounters | 全文外部适用性 |
| D7 算法审计 | 保留为 model sensitivity/audit | 写成替药决策、临床安全或反事实 | Figure 5/6、Abstract、Conclusion |

## 7. 分阶段修改路线图

每一阶段必须在验收完成后再进入下一阶段。尤其不能先润色摘要，再让代码去追逐已经写好的结论。

### Phase 0：冻结基线与建立可复现运行单元

目标：确保后续每个新结果都能回答“由哪份数据、哪次代码、哪组配置生成”。

代码任务：

1. 为正式实验增加唯一 `run_id` 和输出目录，不再向同一 CSV 追加不同运行。
2. 固定 DuckDB pseudo-index、Python、NumPy、PyTorch、DataLoader、bootstrap、采样和绘图随机种子。
3. 输出数据 manifest：源表版本/查询摘要、行数、患者数、阳性数、时间范围、关键文件哈希。
4. 输出训练 manifest：commit、配置、环境、每折患者 ID 哈希、阳性数、checkpoint、temperature、指标。
5. 补全可复现测试依赖，避免混用 Conda 环境的 `site-packages`。

验收标准：同一数据快照和 seed 连续运行两次，标签、pseudo-index、折分和轻量统计完全一致；每份结果都能定位到 commit/config/run。

### Phase 1：重构结局、预测时点和时间边界

目标：消除可以直接修复的目标泄露，并把任务从“看到 onset 化验”改成明确的预测/并发检测设计。

代码任务：

1. 将内部变量逐步从 `dili` 语义迁移为 `ahi_proxy`；为兼容旧产物可暂时保留别名，但新报告必须用 AHI。
2. 明确定义：
   - `t_onset`：首次满足代理阈值的化验时间；
   - `prediction_gap`：预测时点与 onset 间隔；
   - `t_prediction = t_onset - prediction_gap`；
   - 所有 predictors 必须满足 `event_time < t_prediction`。
3. 对阳性病例明确排除 target-defining ALT/AST 事件；检查同一 timestamp 的其他结果是否也属于批量化验泄露。
4. 对阴性病例设计至少三种 pseudo-index 敏感性方案：
   - 固定种子的当前 bounded uniform；
   - 从阳性 onset/住院相对时间分布匹配采样；
   - 固定 landmark/风险窗口设计。
5. 诊断特征改为 encounter-level 且只允许 prediction time 之前已知的信息；若源数据无可靠诊断时间，则主分析删除静态诊断，作为敏感性模型另报。
6. 核验药物时间究竟是 order time 还是 administration time；论文按真实字段命名。
7. 增加契约测试：任何输入事件时间不得等于或晚于 prediction time；目标化验 token/value 不得进入主预测输入。

建议输出：

- `reports/<run_id>/cohort_flow.csv`
- `reports/<run_id>/label_audit.csv`
- `reports/<run_id>/horizon_counts.csv`
- `reports/<run_id>/temporal_leakage_checks.json`

验收标准：随机抽样和全量程序检查均显示零个 post-prediction event；每个 horizon 有明确的 N、positive N、prevalence、观察窗口分布。

### Phase 2：重建患者级训练—选择—校准—测试协议

目标：让模型选择、温度拟合和性能评价各自使用正确的数据角色。

推荐结构：

1. 外层 5-fold `GroupKFold`：只用于最终 evaluation。
2. 每个外层 training fold 内按患者组拆为：
   - model training；
   - selection validation（早停、超参数/epoch 选择）；
   - calibration（只拟合 temperature 或其他 calibrator）。
3. outer test 在模型、epoch、temperature 全部冻结后只评一次。
4. 对同一组 outer-test raw logits 生成 raw 与 calibrated probabilities，构成严格配对比较。

代码任务：

- 修复 [`deep_trainer.py`](../src/diliplus/training/deep_trainer.py) 使用 test AUROC 选 epoch 的逻辑。
- 重构 [`deep_trainer_calibrated.py`](../src/diliplus/training/deep_trainer_calibrated.py) 内部数据角色。
- 将 temperature、model state、split IDs、selection epoch 一起保存。
- early-warning、IG、perturbation 必须通过统一 inference artifact 加载对应 temperature；若解释 raw logit，应明确标记，不混叫 calibrated risk。
- 对 ML baseline 采用同样的数据角色和评价协议。

验收标准：自动测试证明 outer-test ID 不进入训练、早停、超参数、calibration；Figure 3 的 raw/calibrated 每个点来自完全相同的 outer-test logits。

### Phase 3：校准真实模型描述与建立必要消融

目标：让 Methods 描述当前实现，并验证“时间模块是否真的带来增益”。

必须完成：

1. 论文与代码统一为 128 hidden、2-layer/4-head、两个动态流加诊断模态、单任务输出。
2. 时间编码公式改为当前 learnable LogUniformTime2Vec + linear；删除清除、浓度、AUC、pharmacodynamic manifold 等机制词。
3. 修复 Focal Loss：若希望实现 alpha-balanced focal loss，则使用类别相关 `alpha_t`；否则论文只写 focal modulation，不声称类别 alpha balancing。
4. 明确是否保留“MedBERT”名称；没有预训练时写清楚 from-scratch/MedBERT-inspired。

最低消融矩阵：

| 模型 | 用药 | 化验 | 诊断 | 时间编码 | 用途 |
|---|---:|---:|---:|---:|---|
| Demographic/diagnosis baseline | × | × | ✓ | × | 判断静态严重度贡献 |
| Medication-only | ✓ | × | × | 可切换 | 判断用药序列贡献 |
| Lab-only | × | ✓ | × | 可切换 | 判断生化轨迹贡献和泄露依赖 |
| Full without time encoding | ✓ | ✓ | 可选 | × | 检验时间编码增益 |
| Full without diagnosis | ✓ | ✓ | × | ✓ | 检验诊断泄露/静态信息影响 |
| Full model | ✓ | ✓ | ✓或主分析删除 | ✓ | 主模型 |

可选增强：指数衰减时间核、time-aware baseline、ODE 模型。只有在最低消融完成且资源允许时再做，不为满足旧报告措辞盲目扩张。

验收标准：每个模型共享相同 split、输入边界和评价代码；所有结构主张都有相应 ablation 或被降级为实现描述。

### Phase 4：正式重跑核心性能与统计分析

目标：生成一套可以替换论文全部旧数值的唯一正式结果。

必须报告：

- AUROC、AUPRC、Brier、NLL。
- Calibration intercept、slope、reliability plot；QECE 作为补充，明确分箱算法。
- 归一化 pAUC 的 FPR 区间。
- DCA 曲线及预设阈值范围；只有绑定具体低风险临床动作时才解释阈值。
- 每折和 pooled out-of-fold predictions。
- 患者级 bootstrap CI 或合适的重复 grouped-CV uncertainty。
- TA 与最强 baseline 的配对比较；不因均值略高就自动写 statistically superior。

禁止：

- 把 DCA/AUDC 写成 patient benefit 或临床有效性。
- 从单次 calibration 失败推出某方法“fundamentally fails”。
- 在结果出来前预写“ECE 应约为多少”。
- 为保护 TA 故事而贬低在 AUPRC、Brier 或 calibration 上更好的 baseline。

验收标准：Table 2、Figure 2、Figure 3 全部由同一个 run manifest 生成；正文每个数值可回溯到逐样本 OOF prediction。

### Phase 5：重做 early-warning / prediction-gap 分析

目标：把 Figure 4 从有遮蔽语义缺陷的演示，改为严格的 temporal generalisation 分析。

任务：

1. 预注册/预先固定 primary gap；候选 12/24/48/72h 作为敏感性分析。
2. 每个 horizon 报告可评估人数、阳性数、prevalence、序列长度和模态缺失率。
3. 所有模型必须真正消费同义的 cutoff；不能消费 mask 的模型应重新构建截断后的输入，不能简单复用 mask。
4. 静态诊断要么满足时间约束，要么从 early-warning 主模型删除。
5. 使用对应折的 frozen checkpoint 和 temperature。
6. 同时解释 AUROC 与 AUPRC；若 AUROC CI 跨 0.5，不得称 robust。
7. 0h 单独标为 concurrent/onset detection，不与 prospective gaps 混成同一临床主张。

验收标准：对一个手工构造病例，逐 horizon 检查保留事件时间正确；所有模型随 cutoff 改变的输入在语义上等价。

### Phase 6：重做 Table 1、队列图和数据来源描述

目标：先证明研究对象是什么，再讨论模型。

任务：

- 修复 Table 1 SQL/内存中断，生成机器可读 CSV 和论文格式表。
- 报告 unique patients 与 encounters，明确重复住院处理。
- 明确数据库是 ICU、全院住院还是其他场景；给出科室/场景构成。
- Table 1 中将 target group 命名为 AHI-proxy positive，而非 DILI positive。
- 对基线 ALT/AST 的数值和 `abnormal_status` 逻辑做一致性审计。
- 对观察窗口、药物密度、感染/休克等变量同时报告分布和缺失。
- 固定 Figure 1b/1d 采样和 bootstrap seed，记录输入哈希。

验收标准：论文 Table 1 与生成 CSV 逐格一致；样本量、患者数、阳性率在 Abstract、Methods、Figure 1、Table 1 完全一致。

### Phase 7：收缩并验证模型解释与扰动分析

目标：保留有价值的模型审计，彻底切断因果和处方建议暗示。

IG：

- 明确 baseline、积分步数、归一化方法和输出（raw logit 或 calibrated probability）。
- 病例必须来自对应 outer test fold，并使用去标识的 `illustrative encounter`，不把行索引称为 patient ID。
- 正/负 attribution 只写 model-increasing/model-decreasing attribution。
- 增加 cohort-level stability：top features 的 mean absolute IG、跨折一致性或 bootstrap CI。

LOO/Track A：

- 命名为 token deletion / embedding attenuation sensitivity。
- 不把 alpha 解释为剂量比例、停药强度或 withdrawal。
- 正确修改 attention mask；全零 token 现象只用于讨论模型 perturbation 的技术限制。

Track B：

- 让代码读取实际经人工审核的 substitution map；记录 map 版本和审核人/依据。
- 若无临床药师审核，就降级为 predefined token-pair sensitivity examples。
- 结果列使用 `delta_predicted_probability`，不用 ARR。
- 若要声称 manifold preservation，必须预先定义 latent distance/OOD 指标并提供对照；否则删除该主张。
- 全文不得把任何替换称为 safe、equivalent、actionable 或 recommended。

验收标准：图题、轴标签、CSV 列名和正文均不含剂量/因果/安全性误导；任何病例可由固定 fold、checkpoint 和样本索引复现。

### Phase 8：按证据顺序重写论文

论文修改顺序必须晚于核心重跑：

1. Title：使用 AHI proxy / biochemical AHI 和 inpatient 场景；不写 induced、safe、actionable。
2. Abstract：最后写；只放正式 run 的样本量、primary gap 和核心指标。
3. Introduction：临床问题、异步数据问题、预测模型缺口；删除“模型将走向处方决策”的宏大叙事。
4. Methods：数据来源、结局、prediction time、pseudo-index、模态时间边界、group split、模型真实结构、校准、指标、解释方法。
5. Results：先 cohort，再主性能、校准、gap 分析、消融、模型审计；不夹带机制推断。
6. Discussion：principal findings、与既有方法比较、临床解释边界、数据/标签/单中心/时间偏倚/校准限制。
7. Conclusion：只总结预测和模型审计证据，不写 clinical deployment 或 therapeutic alternatives。
8. 按 TRIPOD+AI 核对 title/abstract/methods/results/discussion/open science/subgroup/fairness 报告项。

立即可见但应与系统重写一起处理的错误：删除 Abstract 末尾 `This is TEST.`。

验收标准：用 [`PAPER_CODE_ALIGNMENT.md`](PAPER_CODE_ALIGNMENT.md) 的四问法审查每个方法和结果句；无法定位实现、输入、输出和 commit 的句子删除或标记为 future work。

### Phase 9：最终一致性与投稿前审计

1. 从空环境按 README 重建轻量测试和论文资产。
2. `git diff --check`、契约测试、数据泄露检查、split 检查全部通过。
3. 所有表格、图和正文数字由同一正式 run 生成。
4. 扫描夸大词：`eliminate`、`guarantee`、`unconfounded`、`causal`、`safe`、`actionable`、`optimal`、`superior`、`robust`、`validate/prove`。
5. 核验全部参考文献题名、DOI、年份和正文支持关系。
6. 完成 TRIPOD+AI checklist，并用 PROBAST+AI 自评 participants/data sources、predictors、outcome、analysis 四域。
7. 论文 PDF 编译无错误；只提交 `.tex`、`.bib` 和必要图片，不提交辅助编译产物。

## 8. 优先级任务清单

### P0：不完成就不能相信主要结论

- [ ] P0-01 将研究结局统一为 AHI proxy；决定是否开展临床 DILI adjudication。
- [x] P0-02 修复 target-defining lab 进入动态输入，并固定 24h primary prediction gap；真实数据时序审计 PASS（详见第 15 节）。
- [x] P0-03 修复诊断 encounter/time 边界，或从主模型删除诊断。已采用严格同次住院及 `create_time < prediction_time`，见第 16 节。
- [x] P0-04 核验 ICU/high-acuity 数据来源。Code-08 已证明是全院住院混合 cohort，不是 ICU-only；论文继续使用 inpatient/hospitalised polypharmacy cohort。
- [x] P0-05 删除虚构 MTL/self-calibration，校准真实 128-d/2-layer/4-head 架构和时间公式。论文首轮降调已完成；Code-07 又从 active code 删除 AKI/MTL/tuple 并固定单任务合同，最终稿仍须同步正式新名称和无 alpha 的 loss。
- [x] P0-06 重建 grouped train/selection/calibration/test 协议，修复 test epoch selection。Code-05 已完成并通过真实队列 split 审计。
- [x] P0-07 保存 temperature，并用同一 logits 配对生成 raw/calibrated Figure 3。Code-06 artifact/预测合同已完成；Figure 3 仍待 Code-10 正式 run 重做。
- [x] P0-08 修复 early-warning cutoff 语义。Code-09 已完成三模态物理截断、TextCNN mask 合同、对应折 artifact/temperature 和真实 horizon availability 审计；Figure 4 性能重跑待 Code-10 正式 artifact。
- [x] P0-09 成功重跑 Table 1，明确 patients/encounters/setting。Code-08 已生成 46,864 encounters / 46,844 patients / 391 positives 的 aggregate 证据；论文同步和冻结标签并列项决策仍待完成。
- [ ] P0-10 将 Figure 5/6 降级为模型审计，删除剂量、ARR、安全替换和临床建议暗示。
- [ ] P0-11 生成唯一正式 run，替换论文全部旧数值。
- [x] P0-12 删除 `This is TEST.` 并重写 Abstract/Conclusion。首轮论文文字已完成；正式结果数字仍待 Code-10/12 同步。

### P1：提交前必须完成

- [x] P1-01 固定所有随机种子和 pseudo-index，建立 manifest/run directory。Code-00/04 已完成。
- [x] P1-02 修复/准确描述 Focal Loss alpha。Code-07 改为 `UnweightedFocalLoss(gamma=2)`，接口不接受 alpha，也不使用类别权重。
- [ ] P1-03 最低消融矩阵的 6 个模型合同已完成；训练与 paired comparison 待 Code-10。
- [ ] P1-04 报告 calibration slope/intercept、Brier、NLL、QECE 细节和完整 DCA。
- [ ] P1-05 已报告 24/48/72 h 的 N/positive/prevalence/序列与模态缺失；性能 CI 待 Code-10。
- [ ] P1-06 增加 cohort-level explainability 稳定性分析，或把单病例明确降为 illustration。
- [x] P1-07 核验 MedBERT 命名和是否存在预训练。正式名改为 `TimeAwareMultimodalTransformer`/`MultimodalTransformerBaseline`；确认从头训练，旧 MedBERT 名只作 Python 导入兼容。
- [ ] P1-08 完成 TRIPOD+AI 和 PROBAST+AI 自审。
- [ ] P1-09 系统核验 references.bib。

### P2：增强项，不应阻塞最小可信论文

- [ ] P2-01 时间外验证或多中心外部验证。
- [ ] P2-02 年龄、性别、基础肝病、感染/休克等亚组性能与校准。
- [ ] P2-03 临床专家判定的 probable DILI 子队列敏感性分析。
- [ ] P2-04 指数衰减核、ODE 或其他连续时间模型对照。
- [ ] P2-05 如果未来研究真实换药效果，另立 target trial/因果推断研究，不与本预测稿混写。

## 9. 预计受影响的正式文件

| 任务 | 主要代码/配置 | 主要论文位置 | 需要重跑 |
|---|---|---|---|
| 结局与 prediction gap | `data/labels.py`, `data/sequences.py`, `configs/default.yaml` | Title、Abstract、Methods cohort/outcome、Figure 1/4、Discussion | 全数据、全部模型、全部图表 |
| 诊断时间边界 | `data/diagnoses.py`, `data/dataset.py` | Methods target blinding、architecture、limitations | 全模型 |
| pseudo-index/seed | `data/labels.py`, 新 manifest 工具 | Methods censoring、Figure 1b、limitations | 数据起全部重跑 |
| split/calibration | `training/deep_trainer*.py`, `ml_baselines*.py` | Evaluation、Table 2、Figure 2/3 | 全模型 |
| 架构/时间公式 | `models/diliplus_engine.py`, `models/baselines.py` | Architecture Methods、Discussion | 若改代码则全模型；只校准文字则不重跑 |
| early warning | `evaluation/early_warning.py`, figure script | Figure 4、Abstract、Results、Conclusion | Figure 4 和相关统计 |
| Table 1 | `reporting/table1.py` | Table 1、cohort Results | Table 1/Figure 1 |
| IG/perturbation | `explainability/*.py`, figure scripts | Figure 5/6、auditing Methods/Results | 解释与图 5/6 |
| 文稿夸大 | 无 | 全文 | 不单独触发训练，但必须基于最终结果改 |

## 10. 后续每一步的标准工作方式

以后每次只处理一个清晰单元，并按以下模板记录：

1. **问题 ID**：例如 P0-02。
2. **当前证据**：精确到代码、数据产物和论文段落。
3. **本次决策**：采用哪种设计，为什么。
4. **代码修改**：文件、测试、配置。
5. **运行产物**：run ID、输入、输出、耗时、环境。
6. **结果判断**：是否支持原主张；若不支持就修改论文，不“调参救故事”。
7. **论文同步**：影响的正文、表、图、caption、abstract/conclusion。
8. **Git 记录**：代码 commit、论文 commit、两者对应关系。

建议第一项实际工作从 **P0-02：定义 prediction gap 并修复目标化验泄露** 开始，但在改代码前先用当前数据统计 12/24/48/72h 各 horizon 可用阳性数和事件密度，以此确定 primary gap；不能先凭偏好固定一个数据不支持的窗口。

## 11. 外部方法学依据

本路线图只把这些来源作为规范性依据，不把任何指南替代本项目的实证结果：

- AASLD Practice Guidance on Drug, Herbal, and Dietary Supplement–Induced Liver Injury：DILI 诊断需评估暴露时间、临床/实验室特征并排除更常见竞争病因，不能由单个肝酶阈值自动归因。
  <https://www.aasld.org/practice-guidelines/drug-herbal-and-dietary-supplement-induced-liver-injury>
- TRIPOD+AI（BMJ 2024）：要求透明报告临床预测模型的数据、开发、评价、结果、亚组/公平性和开放科学信息。
  <https://www.bmj.com/content/385/bmj-2023-078378>
- PROBAST+AI（BMJ 2025）：将 participants/data sources、predictors、outcome、analysis 作为预测模型质量、偏倚和适用性审查核心域。
  <https://www.bmj.com/content/388/bmj-2024-082505>
- Guo et al., *On Calibration of Modern Neural Networks*（ICML 2017）：temperature 在 validation data 上优化，并在独立 test data 上评价；temperature scaling 不改变二分类排序，因此同一 logits 上 AUROC/AUPRC 不应因纯温度缩放而改变。
  <https://proceedings.mlr.press/v70/guo17a.html>

## 12. 维护记录

| 日期 | 状态 | 记录 |
|---|---|---|
| 2026-08-16 | 路线图建立 | 完成报告专业性/幻觉校验；以当前代码重写六项核心批评；补入 MTL、架构、诊断、校准、early-warning、Table 1 和解释性分析等报告遗漏；建立 P0/P1/P2 和九阶段执行顺序 |
| 2026-08-16 | 首轮论文文字修订完成 | 按当前代码重写标题、摘要、引言、方法、结果、讨论和结论；删除不存在的 MTL/self-calibration、药代和临床推荐主张；修正架构、指标和提前时距解释；恢复并降调论文 Figure 7 的模型扰动敏感性分析。详见第 13 节 |
| 2026-08-16 | P0-02 完成 | 固定 24h prediction gap；重建 46,864 个住院记录的标签和动态序列；强制 `event_time < prediction_time`；目标 ALT/AST 阈值事件、用药越界和化验越界均为 0；保存独立数据与审计产物。详见第 15 节 |
| 2026-08-16 | P0-03 完成 | 用住院级 `visit_number -> business_uu -> inpatient_f` 桥替代患者级连接；诊断只保留可解析且严格早于 prediction time 的 `create_time`；修复前 30.71% 去重诊断晚于边界，修复后 216,220 条诊断的时间/目标/对齐违规均为 0。详见第 16 节 |
| 2026-08-17 | Checkpoint-01 完成 | 提交 `2bfe6ef` 冻结 Code-00--06 的泄漏安全数据、四方 split、评价与版本化 artifact 合同；checkpoint 验证见第 19 节 |
| 2026-08-17 | Code-07 完成 | 正式模型命名、单任务 AHI-proxy 输出、无权重 focal loss、diagnosis-only modality dropout 和 from-scratch 边界已集中实现；删除 active AKI/MTL/tuple 语义；未训练模型或生成性能数字。详见第 19 节 |

## 13. 已完成：首轮论文文字与 Figure 7 修订（2026-08-16）

### 13.1 修改范围与接手规则

- 论文文件：`D:\PaperWorks\DILI-PLUS\main.tex`。
- Figure 7 制图代码：[`src/diliplus/reporting/figures/perturbation.py`](../src/diliplus/reporting/figures/perturbation.py)。
- 本轮属于“先让论文如实描述当前代码和结果”的文字修订，没有通过新增代码去追逐旧稿主张。
- 后续对话默认以下文字问题已经处理，不应重复扫描或恢复旧措辞。只有正式重跑改变样本量、指标、模型结构或研究设计时，才同步更新对应段落。
- 本轮没有新增 BibTeX 条目。IG 处使用的 `sundararajan2017axiomatic` 已存在于论文 `references.bib`。

### 13.2 已完成的正文修改

| 位置 | 旧稿问题 | 已采用表述/处理 | 修改依据 |
|---|---|---|---|
| Title | 使用 polypharmacy-associated，容易被理解为药物病因归属 | 改为 biochemistry-defined acute hepatic injury、hospitalised patients、time-aware multimodal sequence modelling | 当前标签是 ALT/AST 阈值代理结局，研究场景只能确认到住院记录 |
| Abstract: cohort/outcome | 把队列写成 ICU/DILI | 明确 51,316 hospital encounters、507 AHI-proxy-positive，并说明不是 clinically adjudicated DILI | 与当前标签和数据产物一致 |
| Abstract: censoring | 声称随机右删失消除了观察时间偏倚 | 改为 pragmatic bounded stochastic right-censoring，最多只能减弱观察窗差异 | 当前实现不能证明偏倚被消除 |
| Abstract: performance | 把 0h 和 72h 写成可靠提前预警 | 0h 定义为 concurrent detection；72h 定义为接近随机水平的 stress test | 0h 包含阈值化验；72h AUROC/AUPRC 约为 0.542/0.112 |
| Abstract: interpretation | 写成剂量调整、安全替药和可执行临床建议 | 改为 fitted-model attribution 和 input-perturbation sensitivity | 当前分析只改变模型输入 |
| Introduction | 把生化阳性直接归于多重用药，并扩展到 ICU/critical care | 聚焦生化 AHI proxy、异步事件、类别不平衡和观察窗问题 | 当前数据不支持药物病因或 ICU 泛化 |
| Related Work | 把一般时间嵌入解释为药代动力学 | 改为 longitudinal EHR elapsed-time representation | 代码不含 dose、route、half-life、concentration 或 clearance |
| Cohort/labels | 使用 DILI positive/negative | 统一为 AHI-Proxy Positive/Negative | 避免把自动阈值标签写成临床 DILI |
| K71/diagnosis | 声称 strict target blinding | 描述同次住院、`create_time < prediction_time`、显式目标诊断排除，并披露 create_time 只是数据可用时间代理 | P0-03 已通过真实 Parquet 审计 |
| Prediction time | 把 onset 时点输入称为预测 | 描述固定 24h prediction gap、阳性 onset 与阴性 deterministic pseudo-index；所有输入严格早于 prediction time | Code-02/04 已通过时序与确定性审计 |
| Temporal masking | 把 24/48/72h 遮蔽写成严格前瞻验证 | 明确动态序列被遮蔽但静态诊断未按时点限制 | 当前 Figure 4 只能作为探索性敏感性分析 |
| Architecture | 旧稿写 768 hidden、12 layers、dual stream、预训练式 MedBERT | 改为 hidden size 128；动态流各 2 layers/4 heads；另有静态诊断流；MedBERT-inspired and trained from scratch | 与 `DILIPlusEngine` 和训练配置一致 |
| Time representation | 论文公式和代码不一致，并赋予清除/暴露机制 | 改为 `log(1+Delta t)`、可学习正弦项与线性投影；明确不表示药物浓度或清除 | 与当前 `LogUniformTime2Vec` 实现一致 |
| Training objective | 旧稿声称 AKI 辅助任务、uncertainty MTL、Nash/self-calibration | 删除整个虚构 MTL 叙述，改为 single-task focal loss 和 temperature scaling | 当前没有 AKI label/head，也没有多任务 loss 进入计算图 |
| Focal loss | 暗示固定 0.25 是 class-specific alpha | 明确 0.25 对所有样本相同，不是类别相关 `alpha_t` | 与当前 loss 实现一致 |
| Validation/calibration | 旧稿没有可复核的四方角色 | 描述 patient-grouped training、selection、calibration、outer test 四方隔离；raw/calibrated 来自同一 test logits | Code-05/06 已实现并通过真实 split/artifact 审计；性能仍待正式重跑 |
| Baselines | 架构、hidden size、训练配置和预训练描述不一致 | 按当前 TextCNN、BiLSTM、Transformer 和 MedBERT-inspired baseline 重写表格 | 与正式代码配置逐项核对 |
| Metrics | 把 posterior extremity 当作 calibration，把 DCA 当患者获益 | 分开描述 discrimination、calibration 和 decision analysis；DCA 仅作描述性比较 | 指标含义不能相互替代 |
| Main results | TA/BiLSTM AUPRC 数字错配，并宣称显著优越 | 修正为 TA `0.7389 +/- 0.0402`、BiLSTM `0.6143 +/- 0.0309`；只写 numerically highest，注明未做预设配对显著性检验 | 与当前结果表一致 |
| Calibration results | 声称 TA native self-calibration 最佳 | 明确 TA 的 QECE/AUDC 并非最佳，Figure 3 也不是同一模型严格配对的 before/after | 当前结果和两条训练流水线不支持旧结论 |
| Interpretation case | 把 44190 称为 patient ID，把 attribution 写成 harmful/protective drug | 正文和 caption 明确为 dataset row index；正负号只表示相对参考输入的 model-output direction | 避免身份和效应含义误读 |
| Discussion | 强调可靠预警、药物暴露机制、自校准、临床效用 | 重写为主要结果、当前局限和重建 prediction-gap 数据集的需要 | 结论强度与当前证据一致 |
| Conclusion | 声称可以辅助调整处方 | 只总结代理结局分类和模型行为审计；临床使用需后续独立验证 | 当前研究尚不支持处方建议或临床部署 |
| Test text | Abstract 末尾残留 `This is TEST.` | 已删除 | 非论文内容 |

上表记录的是 2026-08-16 首轮论文修订当时的处理轨迹，不应回写成 Code-07 后的当前代码事实。
尤其是 focal-loss 行中的统一 `0.25` 已在次日的 Code-07 被彻底删除；论文后续必须改为
`UnweightedFocalLoss(gamma=2)`、无 alpha/类别权重，并同步正式模型新名称。Code-07 本身没有
修改论文仓库。

### 13.3 Figure 7：药物 token 扰动敏感性

论文中该图当前自动编号为 **Figure 7**。代码文件仍沿用历史文件名 `Fig_6_InSilico_Simulation.pdf`；LaTeX 图号由前文 `figure` 环境顺序决定，文件名中的 `Fig_6` 不影响编号。

已完成的可见文字替换：

| 旧图文字 | 新图文字 | 原因 |
|---|---|---|
| `Simulated Dose Tapering Trajectories (Patient 44190)` | `Medication-Embedding Attenuation (Row 44190)` | alpha 缩放的是 embedding；44190 是行索引 |
| `Drug Embedding Dilution Coefficient` | `Embedding amplitude multiplier (alpha)` | 不把 alpha 映射为真实剂量 |
| `Predicted DILI Risk Probability` | `Model-predicted AHI-proxy probability` | 与代理结局一致 |
| `Actual` / `Removed` | `Original` / `Zero vector` | alpha=0 不等同于停药或删除 attention token |
| `Simulated Medications` | `Perturbed medication token` | 明确分析对象是输入 token |
| `Alternative Regimen Simulation` | `Predefined Token-Substitution Sensitivity` | 不把 token replacement 写成治疗方案 |
| `Baseline Regimen` / `Simulated Regimen` | `Original input token` / `Substituted input token` | 不暗示临床处方 |
| `ARR = -4.3%` | `Delta p_model = -4.3 pp` | 定义为 substituted-input probability 减 original-input probability，只表示模型输出变化 |

图底部新增边界声明：`Input perturbations quantify fitted-model sensitivity; they are not medication-effect estimates.`

论文正文和 caption 已同时说明：

- Panel A 是 embedding-space attenuation，不是剂量递减或停药；
- Panel B 是 predefined in-code token substitution；
- `Delta p_model` 不是 ARR；
- 结果不证明药物效应、治疗等效性、临床安全性或推荐方案。

验证记录：

- 代码仓库生成图与论文目录图的 SHA-256 均为 `8F6174F54B5DA879C14F783D8A8D3CE2685209738F010247EF6E8484B5328F2B`，确认替换成功。
- `main.pdf` 编译成功，共 11 页；Figure 6 后紧接 Figure 7，编号和顺序正确。
- Figure 7 位于第 10 页，标题、坐标轴、图例、annotation 和 caption 清晰，无裁切、重叠或乱码。

尚未完成的相邻图片问题：论文 **Figure 6（IG waterfall）** 图内仍显示 `Patient 44190`、`Hepatotoxic Contribution`、`Hepatoprotective Contribution` 和 `Prediction Risk Score`。正文与 caption 已经降调，但制图代码 [`src/diliplus/reporting/figures/attribution.py`](../src/diliplus/reporting/figures/attribution.py) 后续仍需将这些可见标签改为 row index、model-output-increasing/decreasing attribution 和 model score。

### 13.4 当前完成状态

| 原优先级项 | 当前状态 | 说明 |
|---|---|---|
| P0-01 结局改称 AHI proxy | 论文文字及 active Dataset/trainer/model 接口已迁移 | 正式产物必须含 `label_ahi_proxy`；构建器暂时额外写兼容别名，临床 adjudication 仍未开展 |
| P0-04 ICU/high-acuity 场景降级 | 论文文字及真实科室审计均完成 | Code-08 证实 hospital-wide inpatient department mix；不得恢复 ICU-only 表述 |
| P0-05 删除 MTL/self-calibration，校准架构描述 | 论文首轮降调、Code-05/06 校准协议和 Code-07 单任务代码均已完成 | 论文仍须同步 Code-07 新正式名及无 alpha loss；性能须正式重跑 |
| P0-10 图示降级为模型审计 | Figure 7 已完成；Figure 6 图内标签待改 | 正文边界已完成 |
| P0-12 删除测试文字并重写 Abstract/Conclusion | 当前结果版本已完成 | 正式重跑后仍需更新最终数字 |

## 14. 后续代码修改大纲（按执行顺序锁定）

后续每次只处理一个 Code Step。完成后必须把修改、测试、产物和论文影响追加到本文件，不能一次混改多个阶段，也不能为保留旧论文结论而调节实现。

| Code Step | 对应优先级 | 目标 | 主要文件 | 最小产物/验收 |
|---|---|---|---|---|
| Image-Fix-01 Figure 6 可见标签 | P0-10 | 在进入数据重构前完成剩余图片降调：把 `Patient 44190`、hepatotoxic/hepatoprotective 和 risk score 改为 row index、model-output attribution 和 model score | `reporting/figures/attribution.py` | 重新生成并替换 Figure 6；PDF 视觉检查确认图内与 caption 一致 |
| **Code-00 基线保护（已完成）** | Phase 0 | 记录当前 commit、配置、数据/结果文件状态，确保现有轻量测试可运行 | `configs/default.yaml`、测试目录、新 manifest 工具 | 当前样本量、阳性数、关键文件哈希和 split 摘要可复核；不覆盖旧结果 |
| **Code-01 prediction-gap 可行性审计（已完成）** | P0-02 前置 | 不先选最好结果，仅统计 12/24/48/72h 各时距可用病例、阳性数、事件数和缺失率 | 新审计脚本，复用 `data/labels.py`、`data/sequences.py` | `horizon_counts.csv`、事件密度/缺失率报告；据此固定 primary gap |
| **Code-02 重建 prediction time 与输入边界（已完成）** | P0-02 | 定义 `t_onset`、`prediction_gap`、`t_prediction`；所有输入严格早于 prediction time；排除 target-defining ALT/AST | `data/labels.py`、`data/sequences.py`、`data/dataset.py`、配置 | 全量 temporal leakage check 为零；0h 与 prospective-gap 数据明确分离 |
| **Code-03 修复诊断 encounter/time 边界（已完成）** | P0-03 | 诊断仅来自同一 encounter 且在 prediction time 前；无法可靠定时则从主模型移除 | `data/diagnoses.py`、`data/diagnosis_audit.py` | 诊断边界契约测试和真实 Parquet 审计均 PASS；without-diagnosis 消融留到 Code-09 |
| **Code-04 固定 pseudo-index 与可复现性（已完成）** | P1-01/P0-02 | 固定 DuckDB、Python、NumPy、PyTorch、DataLoader、bootstrap 和绘图随机种子；实现阴性 pseudo-index 敏感性方案 | `data/labels.py`、配置、manifest/run 工具 | 同一 seed 两次运行标签、pseudo-index、fold 和轻量统计完全一致 |
| **Code-05 重建 grouped split 与 calibration（已完成）** | P0-06/P0-07 | 外层 grouped test；内层 training、selection validation、calibration 分离；test 只评一次 | `training/deep_trainer.py`、`deep_trainer_calibrated.py`、`ml_baselines*.py` | 自动证明 outer-test ID 未参与训练/早停/校准；raw/calibrated 来自同一 logits |
| **Code-06 统一 checkpoint/inference artifact（已完成）** | P0-07 | checkpoint 同时保存模型、split、epoch、temperature、配置和 run ID；下游统一加载 | 新 artifact 工具、training/evaluation/explainability 调用点 | early-warning、IG、perturbation 明确使用 raw 或 calibrated 输出，不再混称 |
| **Code-07 模型和损失语义清理（已完成）** | P1-02/P1-07 | 正式接口迁移到 AHI proxy；移除 focal alpha/类别权重；统一 from-scratch 正式模型命名 | `models/`、training loss、配置、报告标签 | 统一 registry/output helper；语义单元测试覆盖输出维度、loss 公式、时间编码、dropout 和模型命名 |
| **Code-08 重做 cohort/Table 1（代码与真实查询已完成）** | P0-09 | 修复 Table 1 中断；核验 patients、encounters、setting、重复住院和基线肝酶逻辑 | `reporting/table1.py`、cohort audit 工具 | CSV/论文格式、join/schema/baseline audit 与 tracked manifest 已生成；标签并列项决策和论文逐格同步待办 |
| **Code-09 严格 early-warning 与最低消融（合同已完成）** | P0-08/P1-03/P1-05 | 所有模型按相同 cutoff 语义重建输入；完成 medication-only、lab-only、without-time、without-diagnosis、full model | `evaluation/early_warning.py`、模型/训练 pipeline | 24/48/72 h availability 与 6/6 模型语义审计 PASS；性能 CI/paired comparison 待 Code-10 训练 |
| Code-10 正式性能、校准和统计 | P0-11/P1-04 | 生成唯一正式 OOF 结果，完成 AUROC/AUPRC/Brier/NLL、calibration slope/intercept、配对比较和 DCA | training/evaluation/reporting | Table 2、Figure 2/3 和正文数字全部回溯到同一 run manifest |
| Code-11 解释与图片清理 | P0-10/P1-06 | 修正 Figure 6 图内标签；病例限定在 outer test；CSV 改用 row index、embedding attenuation、delta predicted probability；增加 cohort-level 稳定性或明确仅作 illustration | `explainability/*.py`、`reporting/figures/attribution.py`、`perturbation.py` | 图题、轴、图例、CSV、日志和正文术语一致；无 patient ID、dose、ARR 或保护/致病效应暗示 |
| Code-12 正式全流程重跑与论文同步 | Phase 9 | 从冻结数据和配置生成最终表图，更新论文数字并完成投稿前检查 | `pipelines/`、README、paper repo | 单一 run ID；测试和 leakage checks 通过；论文可编译且只提交必要源文件 |

### 14.1 下一步建议

Code-01/02 已于 2026-08-16 完成，见第 15 节；Code-03 见第 16 节；Code-00/04 见第 17 节；Code-05/06 见第 18 节；Checkpoint-01 与 Code-07 见第 19 节；Code-08/09 见第 20 节。进入 Code-10 前先冻结 Code-08 暴露的 baseline ALT/AST 同时间并列项/legacy 标签决策；随后启动唯一正式性能和消融 run。若继续图片清理，仍可独立完成 **Image-Fix-01：Figure 6 可见标签修订**。

## 15. P0-02 执行记录：目标化验与 prediction time 修复（2026-08-16）

### 15.1 本次设计决策

1. 保留旧论文对应的冻结代理结局队列 `51,316 encounters / 507 positives`，本步骤不重新定义 baseline 或 AHI proxy 标签，只修复预测时点和输入边界。
2. 固定首个主任务为 `prediction_gap = 24h`：
   - 阳性 `index_time = t_onset`；
   - 阳性 `prediction_time = t_onset - 24h`；
   - 阴性在 `[first_med_time + 24h, last_med_time]` 内使用 encounter ID 与固定 seed `20260816` 的哈希生成确定性 pseudo-index；
   - 阴性同样使用 `prediction_time = index_time - 24h`。
3. 所有动态事件强制满足 `first_med_time <= event_time < prediction_time`。严格小于号同时排除 prediction time 当刻的整批化验和定义阳性的 ALT/AST 阈值化验。
4. 不覆盖旧 `data_cache/` 产物。修复后数据单独写入 `data_cache/prediction_gap_24h/`；所有模型消费者已切换到 `settings.model_data_dir`，防止旧 onset-time 数据与新数据混用。
5. 本节完成时尚未重训模型。诊断 encounter/time 边界已在随后 P0-03（第 16 节）修复；但 split/calibration、checkpoint 契约完成前仍不得把新数据用于正式性能报告。

### 15.2 修改的正式代码

| 文件 | 修改 |
|---|---|
| `configs/default.yaml`、`src/diliplus/config.py` | 新增 `prediction.gap_hours`、固定 pseudo-index seed、审计 horizons、独立 model-data/audit 目录 |
| `src/diliplus/data/labels.py` | 显式生成 `index_time`、`prediction_time`、`prediction_gap_hours`、`index_time_source`；阴性 pseudo-index 改为确定性哈希；保留 `censor_time=prediction_time` 兼容别名 |
| `src/diliplus/data/sequences.py` | 用药与化验均改为严格 `< prediction_time`；保存 `med_event_times`、`lab_event_times` 以供独立审计 |
| `src/diliplus/data/temporal_audit.py` | 新增真实 Parquet 泄漏审计、目标 ALT/AST 阈值扫描、事件/标签对齐检查和 SHA-256 记录；任何非零违规都会使流水线失败 |
| `pipelines/01_build_dataset.py` | 增加 `temporal_audit` 阶段和 `--stages` 选择性运行接口 |
| training/evaluation/explainability/reporting consumers | 默认数据目录切换为 `settings.model_data_dir`，不再静默读取旧 onset-time Parquet |
| `tests/test_prediction_time_contract.py` 及既有测试 | 增加合法严格边界和泄漏样本的正反契约测试；配置与 pipeline 顺序测试同步更新 |
| `README.md` | 记录 24h 默认任务、局部运行命令、产物目录以及正式重训前置条件 |

### 15.3 可行性结果

来源：`reports/p0_02_prediction_gap_24h/prediction_gap_feasibility.csv`。

| Gap | Eligible positive | Positive retention | Eligible negative | Negative retention |
|---:|---:|---:|---:|---:|
| 0h | 451 | 88.95% | 48,935 | 96.31% |
| 12h | 451 | 88.95% | 48,935 | 96.31% |
| **24h** | **391** | **77.12%** | **46,473** | **91.47%** |
| 48h | 353 | 69.63% | 43,643 | 85.90% |
| 72h | 332 | 65.48% | 40,510 | 79.73% |

24h 被固定为本轮主 gap，不是因为查看模型表现后挑选，而是它在提供明确提前边界的同时保留 391 个阳性；48/72h 留作后续敏感性分析。0h/12h 阳性数相同源于当前化验记录的离散时间结构，不代表两个任务等价。

### 15.4 修复后真实数据与泄漏审计

来源：`reports/p0_02_prediction_gap_24h/temporal_leakage_audit.json` 和 `corrected_dataset_summary.csv`。

| 项目 | 结果 |
|---|---:|
| Audit status | **PASS** |
| Encounters | 46,864 |
| AHI-proxy positive / negative | 391 / 46,473 |
| Prevalence | 0.834329% |
| Medication events audited | 1,370,721 |
| Laboratory events audited | 302,178 |
| Encounters without pre-prediction labs | 6,998 |
| Realised gap mismatch | 0 |
| Prediction time not after first medication | 0 |
| Medication event at/after prediction time | **0** |
| Laboratory event at/after prediction time | **0** |
| ALT/AST value >=120 U/L in model input | **0** |
| Positive index not equal to onset | 0 |
| Missing/duplicate encounter alignment | 0 |

最终产物哈希：

- Labels SHA-256：`21795DED819FF4F8CE6BA4C4E510212372F3B66FC7E71E6EC42DB2D2187FF091`
- Dynamic sequences SHA-256：`9B22E55549A179E9439F8FEA749D6B7C90BF62686E754E04AFC0E22EB79A1E42`

Code-04 为标签行和同一时间戳事件增加稳定排序后，文件字节哈希相对本节初次执行记录发生变化，但 encounter、标签及事件计数均未变化。两次从标签到词表的完整重建已证明上述新哈希保持一致；重新执行 temporal audit 仍为 PASS。

### 15.5 执行中发现并修复的问题

首次真实审计发现：使用 `TO_TIMESTAMP(epoch)` 生成阴性 pseudo-index 会产生 `TIMESTAMP WITH TIME ZONE`，并把同一 `CASE` 中的阳性 onset 也提升为带时区值；与原始本地无时区事件比较时出现约 8 小时偏移。审计因此检测到越界事件并主动失败。

最终实现改为纯 `TIMESTAMP + INTERVAL * fraction` 运算，不再经过 epoch/时区转换。修复后阳性 index 与 onset 全部一致，所有动态事件越界计数归零。该中间失败证明审计不能只检查 SQL 文本，必须检查最终落盘 Parquet 的实际时间值。

### 15.6 测试与当前边界

- `D:\Anaconda\envs\ML311\python.exe -m unittest discover -s tests -v`：16/16 PASS。
- 真实运行命令：`python pipelines/01_build_dataset.py --stages labels sequences temporal_audit`。
- 修复后的结果不能与现有 checkpoint、旧预测 CSV 或论文旧性能数字混用。
- 当前 6,998 个 encounter 在 prediction time 前没有可用肝功能化验；这是后续缺失性分析和 lab-only/full-model 消融必须报告的事实。
- P0-03 已在第 16 节完成；P0-06/P0-07 split/calibration 和正式模型重训仍未完成。

## 16. P0-03 执行记录：诊断 encounter/time 泄漏修复（2026-08-16）

### 16.1 审计结论与严重程度

源模式与视图定义核验确认，可信住院桥为：

`cohort.encounter_id -> analysis.v_patient_encounters.visit_number -> refined.v_discharge_summary.business_uu -> inpatient_f -> refined.v_in_medical_record_diag`

旧实现则先从 medication view 得到患者号，再按 `health_reco` 取回全部诊断，且不检查诊断时间。对修复后的 24h 队列做 aggregate-only 审计得到：

| 修复前审计项 | 结果 |
|---|---:|
| Cohort encounters / 成功建立精确住院桥 | 46,864 / 46,864 |
| 同次住院去重诊断 | 312,700 |
| `create_time` 可解析 | 312,700（100%） |
| prediction time 当时诊断 | 69 |
| prediction time 之后诊断 | 95,947 |
| prediction time 当时或之后合计（过滤前） | 96,016（占全部诊断 30.705469%） |
| 穿过旧目标过滤器的非目标晚诊断 | **95,857（占非目标诊断 30.715817%）** |
| 含 prediction-time-or-later 诊断的 encounters | 29,754 |
| index time/onset 当时或之后诊断 | 88,750；28,441 encounters |
| 阳性中含非目标 prediction-time-or-later 诊断 | **316 / 391 encounters（80.82%）**；1,535 rows |
| 阳性中含非目标 onset-time-or-later 诊断 | **311 / 391 encounters（79.54%）**；1,466 rows |
| 显式 K71/药物性或毒性肝诊断 | 623；其中 464 位于 prediction time 前 |

因此，时间泄漏是已经发生的严重问题，不只是代码风格风险：即使应用旧目标诊断过滤器，旧诊断模态仍在绝大多数阳性 encounter 中看到预测边界之后的信息，且多数阳性可看到 onset 当时或之后录入的非目标诊断。旧 checkpoint、预测 CSV、性能表和基于旧诊断输入的解释结果均不能继续作为修复后模型结果。

关于 encounter 边界需要精确表述：旧代码的患者级连接设计不满足住院级数据契约，但本队列实测的跨 encounter 分配为 0 条；因此不能声称本批数据已发生跨住院污染。旧连接仍产生 774,205 条非目标诊断行，相比同次住院去重的 312,077 条非目标组合多 462,128 条重复/重复分配行。修复仍然必要，因为新实现从结构上保证同次住院，且不依赖本批 ID 恰好未发生碰撞。

### 16.2 本次设计决策

1. 只允许通过上述 `inpatient_f` 住院桥精确关联到同一 encounter，不再使用 patient-only `health_reco` 连接。
2. 以源诊断行 `create_time` 作为**数据库中可用时间的保守代理**；只保留成功解析且严格满足 `diagnosis_time < prediction_time` 的诊断。等于 prediction time 的 69 条也排除。
3. `create_time` 是行政/数据可用时间，不等于疾病真正发生或临床首次识别时间。论文 Methods 和 Limitations 必须如实说明该代理；不得赋予病理发生时间含义。
4. 无可靠时间、边界后诊断以及显式 K71/“药物性肝”/“毒性肝”/“中毒性肝”标签一律排除，不做时间插补。
5. 输出覆盖队列的每一个 encounter；没有合格诊断时保存空列表，避免 merge 时静默丢样本。
6. 诊断按 `diagnosis_time, icd_code, diag_name` 确定性排序并去除完全相同的 code/name/time 组合。

### 16.3 修改的正式代码

| 文件 | 修改 |
|---|---|
| `src/diliplus/data/diagnoses.py` | 删除患者级连接；实现精确住院桥、`create_time < prediction_time`、显式目标诊断排除、稳定去重排序；保存 `diag_event_times`、计数、时间源和契约元数据；异常不再吞掉 |
| `src/diliplus/data/diagnosis_audit.py` | 新增修复前源审计和修复后 Parquet 契约审计；检查 encounter 对齐、重复、列表长度、时间解析、越界和显式目标诊断，并保存 SHA-256；任一违规使流水线失败 |
| `src/diliplus/config.py` | 新增按 prediction gap 隔离的 `diagnosis_audit_dir` |
| `pipelines/01_build_dataset.py` | 新增 `diagnosis_source_audit` 与 `diagnosis_audit` 阶段，严格构建后自动审计 |
| `tests/test_diagnosis_time_contract.py` | 增加合法 prediction-time 前诊断 PASS，以及边界当时 + K71 诊断 FAIL 的正反契约测试 |
| `scripts/run_recorded.py` | 新增持久运行日志/JSON manifest；记录命令、commit、dirty 状态、起止时间、退出码和日志哈希；Windows 子进程固定 UTF-8 |
| `.vscode/tasks.json` | 增加可在 VS Code Terminal/Run Task 中看到的“P0-03 diagnosis build + audit”和 recorded unit tests 任务 |

### 16.4 修复后真实产物与验收

来源：`reports/p0_03_diagnosis_time_gap_24h/diagnosis_tensor_audit.json`。

| 修复后项目 | 结果 |
|---|---:|
| Audit status | **PASS** |
| Cohort / diagnosis rows | 46,864 / 46,864 |
| Encounters with / without eligible diagnosis | 44,119 / 2,745 |
| Retained diagnosis events | 216,220 |
| AHI-proxy positive encounters | 391 |
| Positive with / without eligible diagnosis | 380 / 11 |
| Positive retained diagnosis events | 2,330 |
| Duplicate/missing/out-of-cohort rows | 0 |
| Sequence length mismatch / unparseable times | 0 / 0 |
| Diagnosis at/after prediction time | **0** |
| Explicit target diagnosis in input | **0** |
| 最小 prediction-time margin | 0.00228977 h（约 8.24 s，仍严格早于边界） |

最终诊断 Parquet：`data_cache/prediction_gap_24h/03b_diag_tensors.parquet`，SHA-256 `72EC45E24A795C53ABD1C38D69880A12D2D15F16E776B1ECC6FBA9C3F29F4B52`。Code-04 对同频 token 增加字典序 tie-break 后，诊断词表仍为 5,090 tokens，SHA-256 更新为 `F0C261A03F166E389C2118578880D5746B34BC60CF750F257B0850B1D7845656`；动态词表仍为 16,665 tokens，SHA-256 更新为 `B456824F0ED86C12F04D4EAC33A41008D5BDDB146F51D0D7197D5D4E36A773C2`。这些变化是确定性 ID 分配，不是 token 增删。

### 16.5 可追溯执行记录

| Run ID | 结果 | 内容 |
|---|---|---|
| `p0_03_schema_probe_v1` / `p0_03_diagnosis_views_v3` / `p0_03_mapping_schema_v1` | PASS | 只读核验源表、分析视图定义和精确住院桥 |
| `p0_03_source_audit_v3` | PASS，216.06 s | 复现旧连接，并在应用旧目标过滤器前后分别量化 encounter、prediction time、index/onset 后诊断 |
| `p0_03_diagnosis_build_v1` | PASS，8.87 s | 真实构建严格诊断 Parquet 并完成首轮事后审计 |
| `p0_03_diagnosis_audit_v2` | PASS，3.37 s | 增补阳性分层统计后的最终审计 |
| `p0_03_tests_v4` | **18/18 PASS，2.09 s** | 最终代码状态的全部兼容性、配置、pipeline、动态时间和诊断时间契约测试 |
| `p0_03_vocabulary_v2` | PASS，0.91 s | 基于修复后动态/诊断数据重建词表 |
| `p0_03_dataset_smoke_v1` | PASS，2.27 s | 用正式 `DILIPlusDataset` 加载 46,864 行修复数据与新词表，并取样生成 15-token 诊断张量 |

所有 console 输出保存在 `reports/run_logs/<run_id>.log`，同名 JSON 保存命令、运行环境摘要和日志 SHA-256。中间失败也未删除：`p0_03_tests_v1` 因非必要的 Dataset AST 改动失败，撤回该改动后通过；`p0_03_vocabulary_v1` 因 Windows GBK 无法输出 emoji 失败，记录器改为 UTF-8 后通过。失败留档用于证明没有选择性隐藏执行过程。

### 16.6 当前边界与论文同步要求

- 本步骤没有重训模型，也没有改论文性能数字。旧模型仍使用泄漏诊断，不能迁移或解释为修复后结果。
- Code-05/06 完成后必须从修复数据和新词表训练全部模型；Table 2、Figure 2--7、摘要/结果/结论中的数字按唯一正式 run 更新。
- Code-09 必须至少比较 full model 与 without-diagnosis；2,745 个无预测前诊断的 encounter、以及阳性中的 11 个无诊断 encounter 需要在缺失性/消融解释中披露。
- 论文 diagnosis/target blinding Methods 应从“部分屏蔽、时间边界未修复”更新为本节的精确 encounter + strict pre-prediction availability contract，同时保留 `create_time` 只是数据库可用时间代理的限制。
- 本轮没有 commit、push 或修改论文仓库；Git 提交应在确认下一阶段边界后统一进行。

## 17. Code-00/04 执行记录：基线冻结与可复现性（2026-08-16）

### 17.1 修复前审计结论

Code-02 已把阴性 pseudo-index 改成 encounter ID + seed 的确定性 DuckDB 哈希，因此标签本身不再调用 `RANDOM()`；但项目其余部分仍缺少统一的随机性合同：

- Python、NumPy、PyTorch CPU/CUDA、DataLoader、bootstrap 和绘图分别使用硬编码值、隐式全局状态或未设 seed；
- `GroupShuffleSplit` 使用散落的 `random_state=42`，Logistic Regression/XGBoost 以及 t-SNE/散点 jitter/seaborn bootstrap 没有统一来源；
- 词频相同时依赖 `Counter.most_common()` 的首次出现顺序，事件时间相同时只按时间排序，无法保证数据库并行执行后的字节级顺序；
- `reports/` 被 Git 忽略，虽有本地日志但没有一份可跟踪的轻量基线 manifest；
- 当前代码仓库有尚未提交的 Code-01/02/03 修改，只记录 `HEAD` 不能唯一标识真实执行代码。

因此，修复前只能说 pseudo-index 算法有固定 seed，不能说完整数据构建、折分、训练随机源和论文图片已经可复现。

### 17.2 本次实现

| 文件/范围 | 修改 |
|---|---|
| `configs/default.yaml`、`src/diliplus/config.py` | 新增集中式 `global_seed`、`split_seed`、`bootstrap_seed`、`figure_seed`、确定性 PyTorch 开关、DataLoader worker 数、预注册 pseudo-index 敏感性 seeds，以及 tracked `manifests/` 路径 |
| `src/diliplus/reproducibility.py` | 新增 Python/NumPy/PyTorch/CUDA 统一播种、确定性算法开关、命名空间派生 seed、DataLoader generator/worker 初始化和可序列化 seed manifest |
| training/evaluation/explainability modules | 深度模型、传统模型、内部 split、bootstrap、DataLoader、early-warning、IG 和扰动入口统一读取集中配置；不同 fold/model/stage 使用可复核的命名空间派生 seed |
| Figure 1b/1d 代码 | t-SNE、抽样、散点 jitter 和 seaborn bootstrap 改用 `figure_seed`/`bootstrap_seed` |
| `src/diliplus/data/labels.py`、`sequences.py` | 输出按 encounter 稳定排序；同一时间戳药物用 `med_item`，同一时间戳化验用 `lab_item, lab_value` 作次级排序；序列聚合显式按生成的 `event_seq` 排序 |
| `src/diliplus/data/vocabulary.py` | token 频率相同时按 token 字符串排序，避免首次出现顺序改变 token ID |
| `scripts/run_recorded.py` | 记录 process seed，并在子进程启动前设置 `PYTHONHASHSEED` 和 `CUBLAS_WORKSPACE_CONFIG`；保留命令、commit、dirty 状态、时间、退出码和日志 SHA-256 |
| `src/diliplus/reproducibility_audit.py` | 新增 aggregate-only 基线 manifest、数据/产物哈希、当前 grouped split 摘要、运行时包版本、pseudo-index 敏感性和双重重建比较；不保存 encounter/patient ID |
| `scripts/build_baseline_manifest.py` 等三个入口 | 分别生成基线 manifest、运行 5-seed pseudo-index 敏感性、执行两次完整数据重建并严格比较 |
| `.vscode/tasks.json` | 增加 Code-00/04 确定性重建与 pseudo-index 敏感性任务，运行过程可在 VS Code Terminal 中看到并同步保存日志 |
| `tests/test_reproducibility.py` | 覆盖命名空间 seed 稳定性、Python/NumPy/PyTorch 重复性、GroupKFold 输入顺序不变性和组间零重叠 |

seed 被按职责拆分，避免“改一张图的随机抽样”连带改变 split 或 bootstrap。`PYTHONHASHSEED` 只能在进程启动前生效，所以由 recorded runner 设置；在已经启动的 Python 进程内只记录而不伪称可以回溯修改。

### 17.3 Code-00 冻结内容

tracked manifest 固定保存为 `manifests/code00_code04_baseline.json`，本地副本位于被忽略的 `reports/p0_04_reproducibility/baseline_manifest.json`。它仅含 aggregate counts 与整文件哈希，不写入任何行级标识，记录：

- 基础 commit、dirty 状态、tracked diff SHA-256、每个 untracked 文件 SHA-256 和合成 worktree content SHA-256；
- 配置文件哈希、prediction gap、全部 seed 与确定性开关；
- Python/平台、关键包版本、CUDA 可用性与 GPU 名称；
- 外部只读 DuckDB 的文件名、大小和修改时间；源库过大而未计算全库 content hash，这一限制在 manifest 中显式声明；
- legacy 与修复后关键 Parquet、词表和审计 JSON 的大小与 SHA-256；
- 46,864 encounters、391/46,473 阳性/阴性、三类事件数、时间范围和缺失模态计数；
- 当前 pre-Code-05 五折 `GroupKFold` 的每折数量、阳性数、组数、零 overlap 证明和 membership hash。

当前 worktree 仍是 dirty，基线 manifest 能精确识别本地内容，但不能替代 Git commit。若需要在另一台机器从远端完整重建，仍必须在人工复核后提交这些代码；本步骤没有擅自 commit 或 push。

### 17.4 pseudo-index 预注册敏感性

在未查看模型表现的情况下，配置预先固定 5 个 seeds：`20260816`--`20260820`。每个 seed 只改变阴性 pseudo-index；阳性的观测 onset 不变。审计直接统计严格 prediction-time 前事件，不训练模型，也不据结果挑选 seed。

| Seed | Encounters | Positive / negative | Mean / median observation (h) | Medication events | Laboratory events | Encounters with labs |
|---:|---:|---:|---:|---:|---:|---:|
| **20260816（primary）** | 46,864 | 391 / 46,473 | 99.947975 / 64.297283 | 1,370,721 | 302,178 | 39,866 |
| 20260817 | 46,864 | 391 / 46,473 | 99.620457 / 64.788947 | 1,377,625 | 303,335 | 39,991 |
| 20260818 | 46,864 | 391 / 46,473 | 99.780614 / 64.809658 | 1,378,459 | 302,216 | 39,857 |
| 20260819 | 46,864 | 391 / 46,473 | 96.979585 / 63.973899 | 1,373,865 | 303,202 | 39,854 |
| 20260820 | 46,864 | 391 / 46,473 | 99.058500 / 64.712663 | 1,375,384 | 303,182 | 39,885 |

审核状态 **PASS**：五个 seeds 的队列规模和标签分布完全不变，primary seed 的阴性 pseudo-index digest 与当前标签文件一致（`F18585A3...A850793`）。事件密度会随阴性 observation window 合理变化，因此后续正式模型完成后仍需把这些 seeds 用作敏感性分析，而不能只报告 primary seed。

### 17.5 两次真实全量重建与发现的问题

确定性验证不是只调用 seed 函数。脚本连续两次真实执行：labels → sequences → temporal audit → diagnoses → diagnosis audit → vocabulary，并比较关键产物字节哈希、aggregate dataset summary、五折 membership hash。

第一次运行 `code00_code04_deterministic_rebuild_v1` 用时 455.78 s，审计按设计 **FAIL**：标签哈希一致，但动态序列 Parquet 两次不同。进一步定位为相同 encounter、相同 event time 下缺少稳定 tie-break；数据库返回顺序变化会改变 LIST 顺序，词表同频 token 也存在类似风险。没有掩盖这次失败，也没有降低比较标准。

增加事件次级排序和词表 tie-break 后，`code00_code04_deterministic_rebuild_v2` 用时 424.99 s，最终 **PASS**：

- labels SHA-256：`21795DED819FF4F8CE6BA4C4E510212372F3B66FC7E71E6EC42DB2D2187FF091`；
- dynamic sequences SHA-256：`9B22E55549A179E9439F8FEA749D6B7C90BF62686E754E04AFC0E22EB79A1E42`；
- diagnosis SHA-256：`72EC45E24A795C53ABD1C38D69880A12D2D15F16E776B1ECC6FBA9C3F29F4B52`；
- 两次 comparison payload SHA-256 均为 `C494381F80B9AE00EB858DCDAF0FD3FB1B1A7AFF08E9050328A82C33DEAEFA1D`；
- artifacts、dataset summary、outer split 三个 section 全部完全一致；temporal/diagnosis audits 两次均 PASS。

同一时间戳事件的 tie-break 只确定排列，不改变 46,864 个 encounter、391 个阳性或 1,370,721/302,178/216,220 个药物/化验/诊断事件计数。词表 token 数也仍为 16,665/5,090，但同频 token 的 ID 现在可稳定复现，因此旧 checkpoint 与新词表仍不得混用。

### 17.6 测试、日志与边界

| Run ID | 结果 | 内容 |
|---|---|---|
| `code04_pseudo_sensitivity_v1` | 人工中止 | 初版逐 seed 重扫源表超过 6 min，只有部分日志、无结果 JSON；随后改为一次联合聚合，不作为审计结果 |
| `code04_pseudo_sensitivity_v2` | **PASS，170.64 s** | 5-seed 全量 observation/event-density 敏感性 |
| `code00_code04_deterministic_rebuild_v1` | **FAIL，455.78 s** | 暴露相同时间戳事件顺序的真实非确定性 |
| `code00_code04_deterministic_rebuild_v2` | **PASS，424.99 s** | 修复 tie-break 后两次完整重建逐字节一致 |
| `code00_code04_tests_v2` | **21/21 PASS，3.88 s** | 最终配置、兼容性、pipeline、时间边界、诊断边界与随机性契约测试 |

所有正式 run 的 console 输出保存在 `reports/run_logs/`，成功 run 另有同名 JSON manifest。`code04_pseudo_sensitivity_v1` 因人工中止没有伪造完成 manifest，其部分日志保留。

本步骤没有训练模型，也没有生成新性能数字。当前只能证明：CPU 数据构建、pseudo-index、当时的 pre-Code-05 五折 membership、seed 原语和轻量统计在本环境可复现；不能据此承诺不同 GPU/CUDA/驱动上的深度训练权重逐 bit 一致。Code-05/06 随后已在第 18 节完成。

## 18. Code-05/06 执行记录：四方 grouped split 与版本化模型 artifact（2026-08-16）

### 18.1 修复前审计结论

修复前正式 calibrated 深度训练路径只有三个角色：outer training、一个内部 validation、outer test。内部 validation 同时用于每个 epoch 的早停/模型选择和训练结束后的温度拟合，不能称为独立 calibration set。传统 ML 路径同样把一个内部 holdout 直接用于温度拟合，没有共享的四方 split artifact。

更严重的是旧 `deep_trainer.py` 在每个 epoch 直接评价 outer test，并按 test AUROC 保存所谓最佳 epoch；该未校准 checkpoint 因此发生明确的测试集参与模型选择。Figure 3 又把这套模型与另一套 calibrated 训练结果比较，raw/calibrated 不是同一模型、同一 logits 的配对版本。

Code-06 修复前的 `best_calib_*.pth` 只保存 `state_dict`：不含 fold membership、selected epoch、temperature、配置、数据版本或 run ID。early-warning、IG 和 perturbation 分别重新计算 GroupKFold、加载裸权重并使用 raw softmax；文件名中的 `calib` 并不代表推理应用了温度。

### 18.2 Code-05 四方数据角色

新增 `src/diliplus/splits.py` 作为唯一 split 真值：

1. 外层使用固定 seed 的五折 `StratifiedGroupKFold`，group 是 encounter ID 第一个下划线前的 patient 前缀。
2. 每个 outer-training pool 先划出约 15% calibration，再从剩余 development pool 划出相当于原 outer-training 约 15% 的 selection；其余为 parameter-training。
3. 内层对 128 个固定 seed 的 `GroupShuffleSplit` 候选做确定性选择，要求两侧均有阳性/阴性，并优先使规模和患病率接近母集。
4. 四个角色严格固定：training 只更新参数；selection 只做 epoch/hyperparameter selection；calibration 只拟合温度；test 只进行一次最终 logits 推理。
5. 每折自动检查四角色 index 两两无重叠、patient group 两两无重叠、完整覆盖队列、各角色均含两类；五个 outer test 合并必须恰好覆盖每个 encounter 一次。

深度训练现在只在 selection loader 上早停。选中权重恢复后，calibration loader 只生成一次 calibration logits 并拟合正温度；test loader 随后只生成一次 logits。`y_prob_raw` 和 `y_prob_calibrated` 分别是该同一数组的 `softmax(logits)` 与 `softmax(logits/T)`。固定超参数的 Logistic Regression/XGBoost 保留 selection 分区但不读取它，避免假装执行了不存在的超参数搜索。

旧独立 uncalibrated pipeline 已停用；兼容入口转发到同一个正式协议，`pipelines/02_train_models.py` 只接受 calibrated 模式并要求显式 `--run-id`。因此不能再训练两套不同模型冒充缩放前/后对照。

### 18.3 真实队列 split 审计

来源：tracked `manifests/code05_split_protocol.json` 与本地 `reports/p0_05_code_06/code05_real_data_audit/split_indices.json`。tracked 版本只有 aggregate counts 和 membership hashes，不含 encounter/patient ID；本地版本保存 dataset row indices 供 artifact 复用。

| Fold | Training N / positive | Selection N / positive | Calibration N / positive | Test N / positive |
|---:|---:|---:|---:|---:|
| 1 | 26,236 / 218 | 5,626 / 47 | 5,629 / 47 | 9,373 / 79 |
| 2 | 26,236 / 219 | 5,628 / 47 | 5,627 / 47 | 9,373 / 78 |
| 3 | 26,240 / 219 | 5,625 / 47 | 5,626 / 47 | 9,373 / 78 |
| 4 | 26,235 / 219 | 5,627 / 47 | 5,629 / 47 | 9,373 / 78 |
| 5 | 26,238 / 219 | 5,625 / 47 | 5,629 / 47 | 9,372 / 78 |

五折全部 **PASS**：六种角色对组合在每折的 index overlap 均为 0、patient-group overlap 均为 0；test prevalence 为 0.832178%--0.842846%，selection/calibration 各折均保留 47 个阳性。tracked manifest payload SHA-256 为 `24EE4A1E123F576699B07B5D58FD27DCBAF710C392905B139DE944CFBD12E50D`，并记录 split implementation SHA-256 `27C257F41F6C27DC078A4B920812DEA4756A1F55C22D29760494CFD2BDA0FA6E`。

这次 Code-05 使用新的 stratified outer membership，因此不能与 Code-04 记录的 pre-Code-05 `GroupKFold` membership hash 混用；Code-04 manifest 仍保留为修改前基线，不应被覆盖。

### 18.4 Code-06 artifact 合同

新增 `src/diliplus/artifacts.py` 与 `src/diliplus/calibration.py`。每个正式 artifact 位于 `checkpoints/runs/<run_id>/<model>/fold_XX.pt|joblib`，包含：

- schema version、artifact type、run ID、model name、fold 和创建时间；
- 深度模型 `state_dict`，或 sklearn estimator + 三套已拟合 TF-IDF vectorizer；
- training/selection/calibration/test 的 exact dataset indices 与 aggregate membership summary；
- selected epoch（固定超参数 ML 记为 0）和正 temperature；
- prediction gap、全部 seed、评价协议、训练参数和配置文件 SHA-256；
- 标签/序列/诊断 Parquet 与两套词表的逐文件 SHA-256 和合成 dataset fingerprint；
- 明确的 probability contract：raw=`softmax(logits)`，calibrated=`softmax(logits/T)`，两者共享同一次 test logits；
- metadata payload SHA-256；加载时重新校验 checksum、schema、run/model/fold、四方 split、temperature 和当前数据指纹。

预测与指标改写到 `reports/runs/<run_id>/`，不再向旧 CSV 追加行。训练、评价和解释 pipeline 均要求显式 run ID；early-warning、IG 和 medication-token perturbation 还要求明确选择 `raw` 或 `calibrated`。early-warning 的 test indices 直接来自 artifact，不重新计算折分。IG 候选只能来自 artifact outer test；手工 case 不在该 test fold 时会失败。perturbation 要求 06b 状态中的 run/fold/mode/artifact metadata hash 全部与加载 artifact 一致。

Code-06 同时清除了这些下游代码中的误导标签：`Patient_ID` 改为 dataset `Case_Index`，`Absolute_Risk_Reduction` 改为 `Delta_Predicted_Probability_Points`，embedding attenuation/token substitution 不再写成剂量、反事实或治疗效应。这里只是模型响应描述，不引入因果理论。

### 18.5 验证记录

| Run ID | 结果 | 内容 |
|---|---|---|
| `code05_code06_tests_v2` | **31/31 PASS，4.05 s** | 最终代码状态：原有兼容/泄漏/随机性测试，加四角色隔离、行顺序不变性、温度、deep/sklearn artifact round-trip、metadata checksum/data guard 和 pipeline 参数合同 |
| `code05_real_split_audit_v2` | **PASS，9.11 s** | 最终 split 实现：对 46,864 条真实队列构建五折四方 split；所有 index/group overlap 为 0 |
| `code06_artifact_smoke_v2` | **PASS，10.38 s** | 最终 artifact 实现：用真实数据输入和随机权重 DILIPlusEngine 做保存/加载 smoke；加载前后 logits 逐值相等，raw/calibrated 均由同一 logits 得到 |

artifact smoke 明确使用随机权重，未训练、未计算性能、不得当成实验结果。其本地随机权重 artifact SHA-256 为 `3891D8904B24A9865DE8DEDA43E0BBD5F8631EA5C23E2C418369204CFD983869`，只用于证明序列化与推理合同。

### 18.6 当前边界与下一步

- 本步骤没有正式模型训练，也没有修改论文性能数字。旧 checkpoint、旧 `predictions_calibrated/`、旧 Table 2 和 Figure 2/3 不能解释为新协议结果。
- Figure 2/3 的报告脚本仍读取旧目录；其 run-specific paired-output 重做属于 Code-10，当前不会自动把新旧结果混在一起。
- early-warning 已使用正确 artifact/split/temperature，但逐 horizon 动态 token、lab value 和诊断的 cutoff 合同仍属于 Code-09；当前旧 Figure 4 仍不可复用。
- Code-07 已在第 19 节完成；下一步是 Code-08 重做 cohort/Table 1，随后 Code-09 完成统一消融与 early-warning，Code-10 才启动唯一正式性能 run。

## 19. Checkpoint-01 与 Code-07 执行记录：模型和损失语义清理（2026-08-17）

### 19.1 Checkpoint-01：先冻结已完成合同

在修改模型语义前，先把 Code-00--06 的数据、split、评价和 artifact 合同提交为代码 commit
`2bfe6ef`。该 checkpoint 的目的不是宣布性能有效，而是给后续代码和结果提供可回退、可精确
引用的技术基线。冻结时完成：

- `checkpoint01_tests_v1`：31/31 PASS；
- `checkpoint01_split_audit_v1`：对 46,864 行真实队列的五折四方 split 审计 PASS，tracked
  payload SHA-256 为 `24EE4A1E123F576699B07B5D58FD27DCBAF710C392905B139DE944CFBD12E50D`；
- `checkpoint01_artifact_smoke_v1`：真实格式输入、随机权重的保存/加载 round-trip PASS；没有
  训练、评价模型表现或生成可用于论文的数字；
- `checkpoint01_manifest_v1`：重建 aggregate-only baseline manifest，不保存行级患者/住院标识。

### 19.2 Code-07 前的语义问题

1. `DILIPlusEngine`、`MultiModalBaselineMedBERT` 等名字会让读者误以为模型继承 Med-BERT
   预训练；实际代码没有加载公开 Med-BERT 权重。
2. trainer 中残留 AKI/uncertainty MTL 类和 tuple 兼容分支，但正式模型只有一个分类任务；这些
   死分支使论文夸大的多任务叙述看起来似乎“接近实现”。
3. 旧 focal `alpha=0.25` 对所有样本作统一标量乘法，并不是类别相关 `alpha_t`；它既没有实现
   alpha-balanced focal loss，也没有提供有意义的类别重加权。
4. 模型、trainer、评价、解释和图片代码各自硬编码旧模型键，容易让 artifact 名称与实际类
   语义漂移。
5. 诊断模态随机屏蔽需要明确为逐样本、只作用于诊断表示；不得把它解释为同时缩放药物和
   化验信息。

### 19.3 当前正式合同

| 范围 | Code-07 后合同 |
|---|---|
| 正式主模型 | `TimeAwareMultimodalTransformer`；报告可简称 TA-MMT |
| 正式 Transformer 对照 | `MultimodalTransformerBaseline` |
| 旧 Python 名 | 只作 import compatibility alias；formal registry、训练 CLI、正式流水线生成的新 artifact 和报告标签拒绝旧名 |
| 初始化 | 所有正式深度模型 from scratch；不加载或继承 Med-BERT 权重 |
| 任务/输出 | 单任务 AHI-proxy；统一 helper 要求字典中的 `[batch, 2]` logits；无 AKI head/label、MTL、`log_vars` 或 tuple output |
| 标签 | Dataset 强制要求 `label_ahi_proxy`，通用训练键 `label` 与其同义；只有 `label_dili` 的旧 Parquet 会被拒绝，构建器仅额外写出同值兼容别名 |
| 损失 | `UnweightedFocalLoss`，默认 `gamma=2`，公式为 `(1-p_t)^gamma * cross_entropy`；无 alpha、无类别权重 |
| 模态 dropout | 按样本只清零 diagnosis representation；不连带缩放 medication/laboratory representations |
| 共享配置 | `hidden_size >= 4`、为偶数且能被 `num_heads` 整除，保证四个正式模型均能有效实例化 |
| artifact | 配置快照记录 formal model/loss contract，并纳入 model、baseline、registry、loss、trainer 和 Dataset 实现哈希；历史别名不得成为正式流水线 artifact model name |

主要实现落点为 `src/diliplus/models/registry.py`、`models/diliplus_engine.py`、
`models/baselines.py`、`training/losses.py`、`training/deep_trainer_calibrated.py`、
`data/dataset.py` 和下游 evaluation/explainability/reporting 调用点。兼容 shim 只转发新类，
不保留第二份算法。

### 19.4 验证状态

| Recorded run | 结果 | 验证内容 |
|---|---|---|
| `code07_compileall_v1` | **PASS，0.22 s** | Code-07 涉及的正式 package、pipeline、script 和 tests 通过 Python 语法编译 |
| `code07_tests_v1` | **44/44 PASS，4.38 s** | 全部既有合同与新增模型/损失测试；其中 11 项专门覆盖 focal 手算公式、`gamma=0` 等价 cross-entropy、无 alpha、旧名拒绝、四模型 `[batch,2]` 有限输出、空 mask、时间编码、eval 确定性、diagnosis-only dropout、tuple 拒绝及 legacy-only 标签拒绝 |
| `code07_semantic_audit_v1` | **PASS，2.08 s** | fixed synthetic batch、CPU、4/4 formal models；active MTL/AKI 与预训练加载调用命中均为 0；不读患者数据、不训练、不计算性能 |
| `code07_artifact_smoke_v1` | **PASS，10.59 s** | 用真实格式输入与随机权重 canonical 主模型完成 artifact round-trip；加载前后 logits exact match，raw/calibrated 概率有限且来自同一 logits；不计算性能 |

tracked `manifests/code07_model_loss_contract.json` 的 stable payload SHA-256 为
`67E07E35176C1352E50544D50C968A04EA1E0D39D6849641F701D3D6579BF944`。四个 recorded run 的
console 和命令/环境/退出码/日志哈希记录保存在被 Git 忽略的 `reports/run_logs/`。这些证据证明
结构与执行合同，不是性能证据，也不能替代 Code-10 正式训练。

### 19.5 结果与论文边界

- 本步骤没有正式训练、没有选择模型、没有生成 AUROC/AUPRC/校准/DCA 数字，也没有修改论文
  仓库。所有旧 checkpoint、旧预测、Table 2 和 Figure 2--7 继续标记为 **pre-Code-07**。
- 不能把旧结果中的 TA-MedBERT/MM-BaselineMedBERT 标签机械替换成新名后冒充新实验；新正式
  artifact 必须由修复后数据、formal registry 和唯一 run 重新生成。
- 论文后续必须同步 `TimeAwareMultimodalTransformer`/TA-MMT、
  `MultimodalTransformerBaseline`、from-scratch、单任务 AHI proxy、无 alpha/类别权重 focal
  和 diagnosis-only dropout。该同步只改如实的方法描述；性能数字要等 Code-10 正式 run。
- 下一步是 **Code-08：重做 cohort/Table 1**，再做 Code-09 early-warning/最低消融，最后才进入
  Code-10 正式训练与统计。

## 20. Code-08/09 执行记录：cohort/Table 1、严格 early-warning 与最低消融（2026-08-17）

### 20.1 Code-08 Table 1 查询重构

确认 `pipelines/05_build_paper_assets.py` 的 `table_1` stage 是论文 Table 1 的正式入口，并新增
`--stages table_1`，允许只重建表格而不触发仍基于历史性能数据的图片。

旧实现的主要问题不是一个单独表名，而是把旧标签、人口学、原始用药、未定时诊断和整张
患者级化验表放进同一个大 SQL；这同时造成 legacy/current cohort 混用、潜在跨住院复制、
内存中断和异常被吞掉。新实现以
`prediction_gap_24h/03_dili_dual_stream_tensors.parquet` 为唯一 cohort anchor：

1. 动态事件数和观察窗直接来自模型输入 Parquet；
2. 合并症只来自 Code-03 同 encounter 且早于 prediction time 的诊断 Parquet；
3. 住院和 patient 映射只使用经 schema contract 核验的
   `analysis.v_patient_encounters`；
4. patient profile 仅用于年龄/性别维度；真实性别值 `男/男性/女/女性` 均已映射；
5. 基线化验使用定义该冻结 cohort 的 aligned-lab 缓存，不再二次扫描
   `laboratory_report_sub`；
6. 每个右表在连接前要求目标粒度唯一，连接使用 one-to-one validation；0 匹配、重复键、N
   改变或关系/字段缺失均抛异常并令 recorded run 非零退出。

最终真实运行 `code08-table1-20260817-v3` **PASS（2.79 s）**：

| 项目 | 结果 |
|---|---:|
| encounters | 46,864 |
| unique patients | 46,844 |
| patients with repeated encounters | 20 |
| AHI-proxy positives | 391 (0.8343%) |
| encounter dimension matched | 46,864 / 46,864 |
| patient profile matched | 45,639 / 46,864 (97.3861%) |
| aligned baseline labs matched | 46,864 / 46,864 |
| every join row inflation | 1.000000 |
| critical-care-name screen | 447 encounters |
| other/unknown department | 46,417 encounters |

因此当前队列有数据证据支持“hospital-wide inpatient cohort with department mix”，不支持
ICU-only 表述。年龄和性别均为 45,639/46,864 nonmissing；人口学缺失不改变模型 cohort N。

本地输出位于 `reports/p0_08_cohort_table1/`，包括 machine-readable Table 1、paper text、
cohort summary、department distribution、join audit、schema contract 和 baseline label audit；
tracked `manifests/code08_cohort_table1.json` 只保存 aggregate 和哈希。详细的可复用查询/排错
方法已追加到 `docs/DUCKDB_CLINICAL_DATA_ENGINEERING_PLAYBOOK.md` 第 9.7 节。

### 20.2 Code-08 暴露的标签合同决策

Code-02 为保持旧论文 cohort，使用冻结 legacy label artifact；旧 raw label SQL 对 ALT/AST 只按
`lab_time` 排序，同时间多项没有确定性 tie-break。Code-08 以
`lab_time, lab_item, lab_value` 重建确定性首项后，391 个 positive 中只有 362 个首项状态为
正常/低，且 5 个首项数值 `>=120`；46,473 个 negative 中有 45,260 个首项状态为正常/低。

这不等于已经证明标签错误，因为同一时刻可能同时存在 ALT/AST 多项，且冻结 artifact 的历史
排序选择不可逆；但它证明“冻结标签与当前确定性重建完全一致”不能成立。Code-10 前必须预先
选择并冻结：保留 legacy cohort 并如实披露，或定义同时间多项合并规则后重建全部数据。不得以
哪种方案性能更好作为选择标准。

### 20.3 Code-09 strict early-warning

旧实现把 inter-event delta 反向累加当成事件距 cutoff 的时间，只改变 medication/lab mask，
没有清零当前 `x_med/x_lab/v_lab`，没有截断诊断，TextCNN 也会从被 mask 的卷积窗口读取值。
Code-09 改为 Dataset 显式返回每个 med/lab/diagnosis event 到 `prediction_time` 的真实小时数，
并对三模态统一执行：

```text
event_time < index_time - effective_horizon
等价于
event_age_to_prediction > effective_horizon - base_prediction_gap
```

当前模型数据已在 index time 前 24 h 截断，因此只能评估 effective 24/48/72 h。0 h 和 12 h
所需的 24 h 以后事件已经不在 artifact 中，代码会拒绝这些 horizon，而不是假装重建。每个
被截事件同步清零 mask、token、lab value、delta 和诊断 token；TextCNN 只池化全部 token 均
可见的卷积窗口。evaluation 必须找到四个正式模型的全部五折 artifact，复用其中 test split
与 temperature，并验证五折 test 恰好覆盖每个 encounter 一次；缺 artifact 不再跳过后生成
不完整 CSV。

最终真实运行 `code09-contract-audit-20260817-v2` **PASS（26.51 s）**，未加载 checkpoint、未训练、
未估计性能：

| Effective horizon | N / positive | Med events / missing encounters | Lab events / missing encounters | Diagnosis events / missing encounters |
|---:|---:|---:|---:|---:|
| 24 h | 46,864 / 391 | 1,111,860 / 0 | 283,723 / 6,998 | 215,543 / 2,745 |
| 48 h | 46,864 / 391 | 870,670 / 10,902 | 219,257 / 16,320 | 168,273 / 12,582 |
| 72 h | 46,864 / 391 | 682,260 / 18,986 | 169,350 / 23,455 | 131,732 / 20,323 |

Figure 4 后续除 AUROC/AUPRC 及各自 CI 外，必须同时展示或附表报告这些 availability 数字。
旧 0/24/48/72 性能不再属于当前方法。

### 20.4 最低消融合同

central registry 预注册六项矩阵：

| Artifact name | Medication | Laboratory | Diagnosis | Continuous time encoding |
|---|---:|---:|---:|---:|
| `StaticDiagnosisOnly` | × | × | ✓ | × |
| `MedicationOnly` | ✓ | × | × | ✓ |
| `LaboratoryOnly` | × | ✓ | × | ✓ |
| `FullWithoutTimeEncoding` | ✓ | ✓ | ✓ | × |
| `FullWithoutDiagnosis` | ✓ | ✓ | × | ✓ |
| `TimeAwareMultimodalTransformer` | ✓ | ✓ | ✓ | ✓ |

这里没有 demographic tensor，因此第一项准确称为 static diagnosis-only，而不是虚构的
demographic baseline。所有消融复用 `DILIPlusDataset` 九输入、同一 grouped split builder、
单任务 AHI-proxy、unweighted focal 和 `[batch,2]` output。正式主模型只训练一次；运行
`pipelines/02_train_models.py --run-id <id> --include-ablations` 时，五个 ablation-only 名称与
四个正式架构比较模型进入同一 run，full primary 不会重复。

真实格式合成 smoke 为 6/6 finite logits；禁用模态表示逐值为零；without-time 对任意 dt 改变
保持完全相同输出。tracked manifests 为 `code09_early_warning_contract.json` 和
`code09_minimum_ablation_contract.json`。这些只证明执行合同，不是消融性能结果。

### 20.5 当前边界与下一步

最终验证：

| Recorded run | 结果 | 内容 |
|---|---|---|
| `code08-table1-20260817-v3` | **PASS，2.79 s** | 真实 46,864-row cohort、源 DuckDB schema/join、Table 1 与 aggregate manifest；patient profile 冲突值检查通过 |
| `code09-contract-audit-20260817-v2` | **PASS，26.51 s** | 真实 horizon availability、物理 cutoff synthetic audit、6/6 minimum-ablation smoke |
| `code08-code09-tests-20260817-v2` | **56/56 PASS，4.13 s** | 全套合同测试；新增 0-match/duplicate join、Table 1 missingness、三模态 cutoff、TextCNN mask、temporal metadata 和消融测试 |
| `code08-code09-compileall-20260817-v2` | **PASS，0.14 s** | `src/`、`pipelines/`、`scripts/`、`tests/` 语法编译 |

- 本轮没有修改论文仓库，没有生成新 AUROC/AUPRC、paired comparison 或 Figure 4。
- Code-08 已生成可复核 Table 1，但在论文采用前要先冻结第 20.2 节标签并列项决策；随后逐格
  同步 CSV，不得手工抄用旧表。
- Code-09 的输入和训练入口已具备；performance CI、minimum-ablation paired comparison 和
  Figure 4 属于 Code-10 唯一正式 run。
- Code-10 前仍应提交当前代码/manifest 作为新 checkpoint，避免训练时 dirty implementation
  无法精确引用。
