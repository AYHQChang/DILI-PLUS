# Code-10 正式实验协议

状态：**已预注册；legacy pilot、确定性双重重建、正式 split、Table 1、Code-09 合同和
六模型单种子五折主运行均已通过；128/8 与三种子稳定性仍待运行。** 任何与本协议不一致的
历史 checkpoint、CSV、表格或图片均不能进入论文。

## 1. 研究边界与目标

本实验评价住院多重用药 encounter 中生化定义的 acute hepatic injury（AHI）代理结局风险。
它是回顾性预测任务，不是临床 adjudicated DILI，不推断药物与肝损伤的因果关系，也不直接
证明模型部署会改善患者结局。

主要目标是在相同 cohort、prediction time、输入信息和五折患者分组协议下，公平比较四种
深度架构与两种传统机器学习基线。次要目标评价概率校准、固定告警预算、24/48/72 h
early-warning、最低模态/时间消融和训练随机性。

## 2. 冻结的标签与 cohort 合同

正式标签必须由 `prediction.label_source: deterministic_rebuild` 生成。最早 ALT/AST 时间戳的
全部同行组成基线；只有每一行状态均为正常/低，且全部可解析数值均 `<120 U/L` 时基线才合格。
阳性 onset 是严格晚于基线的首个 ALT/AST 数值 `>=120 U/L`。预测时间为
`index_time - 24 h`，动态事件必须满足 `event_time < prediction_time`。

真实聚合审计 `code10_label_audit_post_rebuild_vscode` PASS：

| 项目 | 数量 |
|---|---:|
| aligned target-lab encounters | 57,643 |
| earliest timestamp 有多行 ALT/AST | 57,496 |
| baseline rule pass / fail | 48,789 / 8,854 |
| raw deterministic positive / negative | 347 / 48,442 |
| 24 h formal encounters | 44,631 |
| 24 h formal patients | 44,611 |
| 24 h formal positive / negative | 315 / 44,316 |
| repeat-encounter excess | 20 |
| missing source patient_id | 0 |

在 legacy 与 deterministic 都合格的 48,789 个 encounter 中标签不一致为 0；cohort 变化来自
更严格且确定性的基线纳入规则。完整 aggregate-only 证据见
`manifests/code10_label_rebuild_audit.json`。legacy 标签只能在 `pilot_legacy` run 中验证流水线，
不得被 finalizer 接受为正式结果。

## 3. 数据拆分

外层为五折 `StratifiedGroupKFold`。分组键必须是
`analysis.v_patient_encounters.patient_id`，禁止再从 encounter 字符串猜测 patient。每个外层
development pool 再划分 parameter-training、selection 和 calibration：

- training：只更新模型参数、拟合 TF-IDF/其他预处理；
- selection：只选择 epoch 或预注册超参数；
- calibration：只拟合正标量 temperature；
- test：模型、epoch、预处理和 temperature 冻结后，只收集一次 logits。

同一患者不能跨任意两个角色。五个 outer-test 折必须恰好覆盖 cohort 一次。

## 4. 六个主比较模型

| 模型 | 正式角色 | 共同合同 |
|---|---|---|
| `MultiModalTextCNN` | convolutional deep baseline | 128-d 融合、相同输入/损失/训练预算 |
| `MultiModalBiLSTM` | recurrent deep baseline | 128-d 融合、相同输入/损失/训练预算 |
| `MultimodalTransformerBaseline` | 无连续时间增强 Transformer | 128-d、2层、4头 |
| `TimeAwareMultimodalTransformer` | primary model | 128-d、2层、4头 |
| `LogisticRegression` | linear TF-IDF baseline | 无 class weight、训练集拟合词表 |
| `XGBoost` | nonlinear tree baseline | 无 class/sample weight、固定参数 |

所有深度模型从随机初始化开始，目标为单任务 AHI proxy，使用无类别权重 focal loss
（`gamma=2`）。不加载 Med-BERT 或其他预训练权重，不包含 AKI/MTL/self-calibration。
不强制不同架构精确匹配参数量；每个模型必须报告参数量或特征量、训练时间、峰值 GPU 显存和
推理设备。

主配置是 `configs/default.yaml` 的 128-d/4-head。`configs/sensitivity_128d_8heads.yaml` 只改变
Transformer head 数为 8；它是预注册架构敏感性，不用于看过 outer-test 后选择主配置。

## 5. 训练、选择和校准

- 正式深度模型最多 50 epochs，Adam、学习率 `1e-4`、weight decay `1e-5`；
- scheduler 和 early stopping 唯一监控 selection AUPRC；patience 分别为 3 和 7；
- 每个 fold 保存逐 epoch loss、selection AUPRC/AUROC、学习率和 selected epoch；
- calibration split 拟合 temperature `T>0`；
- 同一 test logits 生成 `softmax(logits)` 与 `softmax(logits/T)`；
- 单个 model/fold 内，标量温度不改变排序；fold-level raw/calibrated AUROC/AUPRC 仅允许
  `1e-5` 内的浮点并列差异。各折温度不同，因此 OOF 合并后跨折排序可能变化，pooled raw 与
  calibrated discrimination 不要求相等；两者必须同时保存并明确概率模式；
- 主六模型先完成单 seed/5-fold。确认完整性后，仅对 primary 和最强深度 comparator 追加
  3-seed 稳定性分析；“最强”按主 run 的 pooled calibrated AUPRC 定义，选择规则在读取结果前
  已写入本协议。

## 6. 评价指标

### 6.1 主指标与判别

- 唯一模型选择主指标：AUPRC/average precision；
- 报告 prevalence 随机基线与 `AUPRC / prevalence` lift；
- AUROC 为关键次要指标；
- normalized partial AUC：FPR `<=0.05/0.10/0.20`。

### 6.2 校准与整体概率误差

- Brier score 和以各 fold training prevalence 为参照的 Brier skill score；
- negative log-likelihood；
- calibration intercept、slope、observed/expected ratio；
- 十等频 calibration bins/plot；
- quantile ECE 仅作补充，不作为“校准良好”的单独证据。

### 6.3 固定阈值、告警预算和决策曲线

- calibrated risk thresholds：0.5%、1%、2%、5%；
- 每个阈值报告 sensitivity、specificity、PPV、NPV、F1、balanced accuracy、MCC 和 alert N；
- top 0.5%、1%、2%、5% 告警预算报告 precision、recall 和 alert N；
- DCA 阈值范围 0.5%--5%，步长 0.25%，同时保存 model/treat-all/treat-none net benefit；
- DCA 只描述给定阈值下的决策分析，不写成已证明临床获益。

### 6.4 不确定性与模型比较

- 五折 OOF 预测按 dataset index 对齐后合并；
- 以真实 patient_id 为 cluster 做 1,000 次 bootstrap；
- AUROC、AUPRC、Brier、NLL 报告 pooled estimate 与 95% CI；
- primary 与五个 comparator 使用相同 bootstrap resample 做配对差异；
- pairwise two-sided bootstrap p value 使用 Holm 校正；
- Brier/NLL 的 `primary - comparator < 0` 表示 primary 更好，AUROC/AUPRC 则相反；
- 不仅报告 p value，还必须报告差值和 95% CI。

## 7. Early-warning、消融和敏感性

24/48/72 h early-warning 必须加载主 run 对应 fold checkpoint 和 temperature，不能按 horizon
重新训练或校准。每个 horizon 同时报告样本/阳性数、三模态可用性、AUPRC/AUROC、Brier/NLL
和校准 slope。12 h 不能由24 h截断 artifact 重建，当前不进入正式分析。

最低消融使用 Code-09 预注册的 diagnosis-only、medication-only、laboratory-only、full without
time、full without diagnosis 和同一 full primary。主模型不重复训练；消融与 full model 用相同
split/seed/optimizer/loss，并做配对 OOF 比较。128/8 head 分析和后续 3-seed 分析均单列为
sensitivity，不与六模型主表混合。

## 8. 运行顺序与停止条件

1. 聚合标签审计；
2. legacy 1-epoch/1-fold pilot，验证 GPU、显存、速度和全部输出字段；
3. 确定性重建 labels、sequences、diagnoses、vocab 和时间/split审计；
4. 从 clean Git commit 启动六模型单 seed/5-fold主 run；
5. 汇总 OOF、cluster bootstrap、配对比较、calibration/DCA；
6. 128/8 sensitivity；
7. primary + 主 run 最强 deep comparator 的 3-seed stability；
8. 最低消融和 early-warning；
9. 最后才重建论文 Table 1/2 和 Figures 2--4。

任一 data lineage、patient grouping、split overlap、test coverage、artifact checksum、单次 logits、
NaN/Inf、缺 fold 或模型问题都会令 run 状态为 FAIL；不得用部分结果制图。

## 9. 保存与隐私

本地 `reports/runs/<run-id>/` 保存：逐 fold logits/raw/calibrated probability、history、fold metrics、
pooled metrics、clustered CI、paired comparison、calibration bins、DCA curve、resource usage 和
`run_manifest.json`。`checkpoints/runs/<run-id>/` 保存模型、split indices、temperature、配置和
数据指纹。患者级文件由 `.gitignore` 排除。

`manifests/code10_formal_run.json` 只保存 aggregate 结果、文件哈希、commit/config/data fingerprint、
模型/fold完整性和硬件环境，不保存 patient/encounter ID。该 aggregate manifest 是后续绘图、
制表和论文数值的唯一追踪入口。

## 10. 参考方法（写入论文前由作者核验并加入 BibTeX）

- Saito T, Rehmsmeier M. *The Precision-Recall Plot Is More Informative than the ROC Plot When
  Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS One, 2015.
- Guo C et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
- Vickers AJ, Elkin EB. *Decision Curve Analysis: A Novel Method for Evaluating Prediction Models.*
  Medical Decision Making, 2006.
- Collins GS et al. *TRIPOD+AI statement.* BMJ, 2024.
