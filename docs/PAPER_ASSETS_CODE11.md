# Code-11：正式 Table 2 与 Figures 1–5

## 1. 状态与边界

Code-11 已把 Code-10 的正式 aggregate 结果转换为论文表图。它不重新训练模型、不重新拟合
temperature、不连接 DuckDB，也不读取患者级原始数据。所有数值均来自已冻结的 run-specific
OOF 评价文件；绘图只改变表达，不改变任何统计估计。

正式生成入口：

```powershell
D:\Anaconda\python.exe scripts\run_recorded.py `
  --run-id code11_reframed_paper_assets_v2 `
  --seed 20260816 -- `
  D:\Anaconda\python.exe pipelines\05_build_paper_assets.py `
  --stages table_2 figure_1 figure_2 figure_3 figure_4 figure_5 asset_manifest
```

当前重构版的最终 recorded run `code11_reframed_paper_assets_v2` PASS（11.54 s）。本地日志位于
`reports/run_logs/`；可提交的 aggregate-only 输出合同位于
`manifests/code11_paper_assets.json`。

全量回归测试 `code11_reframed_full_tests_v1` 为 68/68 PASS；它验证了队列连接、prediction-time、
诊断时间边界、模型/损失、split/artifact、Code-08/09/10指标以及新主文图序列，未启动训练。

## 2. 唯一数据来源

| 论文资产 | 正式来源 | 统计合同 |
|---|---|---|
| Figure 1 | `code10_label_rebuild_audit.json`、`code05_split_protocol.json`、`code09_early_warning_contract.json` | 确定性队列、严格24 h prediction-time边界、三模态可用性、五折患者分组角色；只读aggregate manifests |
| Figure 2 | Code-08 `table1_characteristics.csv` + tracked cohort/Table 1 manifest | 仅使用聚合的中位数、四分位数和SMD；强调观察机会和记录强度，不作机制解释 |
| Table 2 / Figure 3 | `code10_formal_128d4h_seed0/metrics/` | 六模型 calibrated pooled OOF；1,000 次 patient-cluster bootstrap；校准与固定告警预算 |
| Figure 4 | Code-09 availability manifest + 正式 run `early_warning/metrics/` | 同 checkpoint/temperature/test membership；strict 24/48/72 h；无重训/重校准 |
| Figure 5A | `code10_minimum_ablations_seed0/metrics/` | full-minus-ablation 配对 cluster bootstrap、Holm 校正 |
| Figure 5B | `code10_sensitivity_128d8h_seed0/metrics/` | 128d/8-head minus 128d/4-head；matched seed/splits |
| Figure 5C | `code10_seed_stability_summary/` | TextCNN 与 primary 的 matched seed 0/1/2；mean ± sample SD 仅作优化随机性敏感性 |
| Figure 5D | 正式 run `resource_usage.csv` | 四个深度模型的参数量、单折平均时长和峰值GPU显存；描述计算代价而非临床效用 |

`src/diliplus/reporting/formal_assets.py` 是结果读取、正式模型显示名和颜色的唯一注册表。
加载器会检查 formal manifest 状态、run ID、六模型集合、概率模式、必要列，以及主 run
metric 文件哈希；不会自动搜索旧 `reports/predictions*` 或历史 early-warning CSV。

## 3. 固定模型色卡

| Model | Display label | Colour |
|---|---|---|
| `LogisticRegression` | Logistic regression | `#FFB3DD` |
| `XGBoost` | XGBoost | `#999ACD` |
| `MultiModalTextCNN` | TextCNN | `#F8BF92` |
| `MultiModalBiLSTM` | BiLSTM | `#99CDCE` |
| `MultimodalTransformerBaseline` | Multimodal Transformer | `#99BADF` |
| `TimeAwareMultimodalTransformer` | TA-MMT | `#DF9E9B` |

同一模型在所有主图、后续补充图和图例中必须保持同色。消融变体使用单独、固定的消融色卡；
full TA-MMT 仍使用 `#DF9E9B`。颜色不是优劣编码，所有模型线宽和点大小原则上相等。

## 4. 统一版式合同

正式 Figures 1--5 统一使用 Times New Roman（不可用时只回退 DejaVu Serif）、14号基础字、
16号加粗面板标题、加粗坐标与图例，以及600 dpi PNG和可编辑文字PDF。模型色卡仍由
`formal_assets.py` 唯一注册；字体变化不改变任何数据、统计量、排序或结论。

## 5. Figure 1：队列与预测时点合同

生成器：`reporting/figures/study_design.py`。

- A：57,643条目标ALT/AST记录经确定性baseline规则和24 h历史要求形成44,631 encounters；
- B：明确 `event time < prediction time = index time - 24 h`，盲区内信息不进入模型；
- C：展示正式24 h时距下药物、化验、诊断的保留事件量与非空率；
- D：展示training/selection/calibration/outer test四个患者互斥角色及五折范围。

Figure 1只读取三个已跟踪aggregate manifest，不访问DuckDB或患者级数据。它取代旧Figure 1b
和1d；旧图的legacy标签、任意轨迹表达和过时队列数字不再进入默认论文资产流水线。

## 6. Table 2

生成器：`src/diliplus/reporting/table2.py`。

输出包括：

- numeric CSV：保留 point estimate、CI上下界、run ID 和 probability mode；
- paper CSV/TXT/LaTeX：供人工核对和论文排版；
- AUROC、AUPRC 的 95% CI 来自 patient-cluster bootstrap；
- AUPRC lift 以事件率为参考；
- Brier、NLL、calibration slope、O:E 用于区分 discrimination 与 calibration；
- top-1% PPV/sensitivity 是固定告警预算，不是事后选择概率阈值。

Table 2 不把 raw 与 calibrated 当作两个独立训练模型。正式表使用 calibrated OOF 概率；
fold-specific temperature 可改变跨折 pooled 排序，因此 pooled raw/calibrated discrimination
不要求完全相同，但每折内部排序不变。

## 7. Figure 2：队列观察与测量过程

生成器：`reporting/figures/observation_process.py`。

- A：阳性与阴性 encounter 在 prediction time 前的观察窗中位数和IQR；
- B：药物、化验和诊断的记录事件负担；
- C：baseline ALT、AST、总胆红素的记录值分布；
- D：Table 1 中绝对SMD最大的变量，突出记录强度和观察机会差异。

Figure 2 将论文从“模型竞赛”转向“数据生成/观察过程”。它只能说明两组在可见历史长度、事件
密度和基线测量上存在差异；不能把这些差异解释为疾病机制、药物作用或因果效应。数值读取时
强制核对 Code-08 manifest 中的 Table 1 SHA256，避免再次误接旧表或错误 cohort。

## 8. Figure 3：紧凑预测基准

生成器：`reporting/figures/calibration_impact.py`。

- A：六模型 calibrated AUPRC 与 patient-cluster bootstrap 95% CI，虚线为事件率；
- B：校准斜率与 observed:expected，理想值均为1；
- C：固定 top-1%（447 encounters）告警预算内捕获的AHI-proxy事件数和其余告警数。

Figure 3 保留模型结果，但只展示支撑结论所需的三个互补维度，避免在多幅主图中重复比分。
它清楚显示简单模型在罕见事件判别、概率一致性和固定资源预算下不弱于复杂序列模型；不把
统计比较写成临床有效性或可部署性证明。

## 9. Figure 4：更早截止点的信息侵蚀

生成器：`reporting/figures/early_warning.py`。

- A：相对24 h，48/72 h仍保留的药物、化验和诊断事件比例；
- B：各时距仍有非空药物、化验或诊断输入的 encounter 比例；
- C：每种模态的中位 retained token 数及IQR band；
- D：四个深度模型的 matched AUPRC stress-test轨迹和patient-cluster CI。

48/72 h 使用24 h训练模型和既有 temperature，没有按 horizon 重校准。因此更早时距的概率误差
和AUPRC下降必须与A--C的信息可用性下降一同解释：这是 input-availability stress test，不是
各时距重新优化后的性能上限，也不能单独归因于时间编码或某一模态。

## 10. Figure 5：模型复杂度审计

生成器：`reporting/figures/robustness.py`。

- A：full-minus-ablation 的配对AUPRC差值；负值表示消融版本更高，星号表示Holm-adjusted
  `P<0.05`；
- B：两个Transformer的8-head-minus-4-head配对AUPRC差值；
- C：TextCNN/TA-MMT三个matched seeds，灰线连接相同seed，黑色横线/误差棒为mean ± sample SD；
- D：四个深度模型的可训练参数量、平均fold时长和峰值GPU显存。

三 seed 是训练随机性敏感性，不是三个独立队列，不计算伪造的跨 seed CI。Figure 5 支持的
主要结论是 medication-only 的AUPRC高于full primary、time encoding没有可检测的增量、
8 heads未显示稳定改进、primary的seed波动更大，且额外复杂度带来显著计算成本；这些均是
预测/模型敏感性描述，不是因果结论。

## 11. 单病例审计图边界

`attribution.py` 和 `perturbation.py` 已统一Times New Roman版式、改用明确的Figure 6/7文件名，
并移除patient ID、hepatotoxic/hepatoprotective、risk score、剂量/治疗方案等可见措辞。由于当前
`06b/06c` CSV来自pre-Code-07历史模型，这两图不在默认构建和formal manifest中；必须先用正式
run、固定outer-test输入和当前artifact合同重跑解释流水线，才可生成论文补充图。

## 12. 运行时故障与修复记录

前两次 recorded run `code11_formal_paper_assets_vscode` 和 `_v2` 均在 ML311 的首次 PNG
`savefig` 触发 Windows `0xC06D007F`，退出码 `3228369023`。最小测试进一步确认：Matplotlib
import、`pyplot` 和 `subplots` 均成功，PNG 写出时崩溃。失败发生在渲染依赖，不涉及数据加载、
统计、训练或 checkpoint。

最终采用机器上已有的 `D:\Anaconda\python.exe`（Python 3.13 / Pandas 2.2.3 /
Matplotlib 3.10.0，Agg backend）只执行 aggregate CSV→PNG/PDF 渲染。最小 PNG/PDF smoke PASS；
最终 run v4 PASS。渲染环境完整写入 `code11_paper_assets.json`，没有安装或修改任何环境。

## 13. 旧图片清理、写作大纲与论文同步

主文新文件名固定为 `Fig_1_Study_Design`、`Fig_2_Observation_Measurement_Process`、
`Fig_3_Compact_Predictive_Benchmark`、`Fig_4_Earlier_Cutoff_Information_Erosion` 和
`Fig_5_Model_Complexity_Audit`。被替代的旧版正式Figure 2--5（PNG/PDF共8个）已在视觉验收后删除，防止误复制；
这些生成物可由旧生成器重建，但不再由默认论文资产流水线调用。
论文新叙事和逐节写作边界已保存到论文仓库 `PAPER_REWRITE_OUTLINE.md`。

代码仓库完成不等于论文已同步。下一阶段必须人工核对 caption 后，将最终 PDF 复制到论文
仓库、更新 `main.tex` 的文件名/图号/结果文字，并把原单病例 attribution/perturbation 降为
补充性 model audit。当前 Code-11 没有修改论文仓库。
