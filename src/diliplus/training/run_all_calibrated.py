"""
DILI-PLUS | 校准模型批量训练入口（包实现）

职责：依次调用 calibrated 深度训练器完成四个深度模型，再调用传统 ML 训练器
完成 Logistic Regression 与 XGBoost；两类模型均使用温度缩放而非等渗回归。
输入：已准备好的 Parquet 与词表。
输出：六种模型的校准预测、权重和汇总结果表。
状态：当前主训练编排器；不执行数据构建、早期预警、解释或制图，因此不是完整的
端到端流水线。
"""

from diliplus.config import load_settings
from diliplus.training.deep_trainer_calibrated import main as train_deep_model
from diliplus.training.ml_baselines_calibrated import main as train_ml_models

def run_experiments(settings=None):
    settings = settings or load_settings()
    
    dl_models_to_train = [
        'MultiModalTextCNN', 
        'MultiModalBiLSTM', 
        'MultiModalBaselineMedBERT',
        'MultiModalTimeAwareMedBERT' # 即我们的 DILI-PLUS 引擎
    ]
    
    print("🚀 [GLOBAL PIPELINE] Starting Ultimate Calibrated Experiment Pipeline...")
    print(f"📋 Target DL Models (4): {dl_models_to_train}")
    print("📋 Target ML Models (2): ['LogisticRegression', 'XGBoost']\n")
    
    # =========================================================
    # 阶段一：运行 4 个深度学习校准模型
    # =========================================================
    for model_name in dl_models_to_train:
        print("="*70)
        print(f"🟢 [LAUNCHING DL PROCESS] Calibrating DL Model: {model_name}")
        print("="*70)
        
        train_deep_model(["--model", model_name], settings=settings)
        print(f"\n✅ [SUCCESS] Finished DL Model: {model_name}.\n")

    # =========================================================
    # 阶段二：自动调用 2 个机器学习基线
    # =========================================================
    print("="*70)
    print(f"🟢 [LAUNCHING ML PROCESS] Training Machine Learning Baselines")
    print("="*70)
    
    train_ml_models(settings=settings)
    print(f"\n✅ [SUCCESS] Finished ML Baselines.\n")
        
    print("\n🎉🎉🎉 GLOBAL PIPELINE COMPLETE: All 6 Models Successfully Trained & Calibrated! Ready for Visualizations! 🎉🎉🎉")

if __name__ == "__main__":
    run_experiments()
