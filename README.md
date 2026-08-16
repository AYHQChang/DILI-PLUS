# DILI-PLUS

DILI-PLUS 是一个面向住院多重用药场景的 DILI 风险建模项目。实现代码采用 `src/` 包布局；数据构建、训练、评估、解释和论文产物分别由五个阶段入口组织。

## 推荐运行方式

```powershell
conda activate ML311
python pipelines/01_build_dataset.py
python pipelines/02_train_models.py --mode calibrated
python pipelines/03_evaluate_models.py
python pipelines/04_explain_models.py
python pipelines/05_build_paper_assets.py
```

全部入口均接受 `--config configs/default.yaml`。项目刻意不提供默认“一键跑完”命令，以避免误触发数据库提取、长时间训练或批量覆盖图片。

## 目录职责

```text
configs/          集中路径与训练默认值
pipelines/        五个可执行阶段入口
src/diliplus/     唯一的当前实现
tests/            无数据库、无训练、无绘图的契约测试
archive/legacy/   不进入新流水线的历史代码
```

根目录中的数字编号脚本是兼容入口，用来保证旧命令、实验笔记和批处理不失效。它们不包含第二份算法，也不会重复执行计算；新工作应直接使用 `pipelines/`。

## 产物兼容合同

默认配置仍将产物写入原目录：Parquet 到 `data_cache/`，词表到 `vocab/`，权重到 `checkpoints/`，预测与研究表格到 `reports/`，PNG/PDF 到 `figures/`，药物映射 CSV 留在项目根目录。相对路径始终以项目根目录解析，与启动时的当前工作目录无关。

源医疗 DuckDB 只能经 `diliplus.database.connect_source_database` 连接，并固定使用 `read_only=True`；配置文件不能将其改为写模式。
