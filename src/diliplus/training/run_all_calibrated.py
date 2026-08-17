"""
DILI-PLUS | 校准模型批量训练入口（包实现）

职责：默认完成四个正式深度模型和两个传统 ML；显式启用时，在同一 run/split
合同下额外训练五个预注册最低消融。两类模型均使用温度缩放而非等渗回归。
输入：已准备好的 Parquet 与词表。
输出：六种模型的校准预测、权重和汇总结果表。
状态：当前主训练编排器；不执行数据构建、早期预警、解释或制图，因此不是完整的
端到端流水线。
"""

import argparse

from diliplus.config import load_settings
from diliplus.models.registry import ABLATION_ONLY_MODEL_NAMES, FORMAL_DEEP_MODEL_NAMES
from diliplus.training.deep_trainer_calibrated import main as train_deep_model
from diliplus.training.ml_baselines_calibrated import main as train_ml_models

def run_experiments(run_id, settings=None, include_ablations=False):
    settings = settings or load_settings()
    
    dl_models_to_train = list(FORMAL_DEEP_MODEL_NAMES)
    if include_ablations:
        dl_models_to_train.extend(ABLATION_ONLY_MODEL_NAMES)
    
    print("[DILI-PLUS] Starting the versioned single-task training pipeline...")
    print(f"Target DL Models ({len(dl_models_to_train)}): {dl_models_to_train}")
    print("Target ML Models (2): ['LogisticRegression', 'XGBoost']\n")
    
    # =========================================================
    # 阶段一：运行 4 个深度学习校准模型
    # =========================================================
    for model_name in dl_models_to_train:
        print("="*70)
        print(f"[LAUNCHING DL PROCESS] Calibrating DL Model: {model_name}")
        print("="*70)
        
        train_deep_model(
            ["--model", model_name, "--run-id", run_id], settings=settings
        )
        print(f"\n[SUCCESS] Finished DL Model: {model_name}.\n")

    # =========================================================
    # 阶段二：自动调用 2 个机器学习基线
    # =========================================================
    print("="*70)
    print("[LAUNCHING ML PROCESS] Training Machine Learning Baselines")
    print("="*70)
    
    train_ml_models(["--run-id", run_id], settings=settings)
    print("\n[SUCCESS] Finished ML Baselines.\n")
        
    print("\nGLOBAL PIPELINE COMPLETE: all 6 models trained and calibrated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all formal DILI-PLUS models")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_experiments(args.run_id)
