"""
DILI-PLUS | 未校准深度模型批量训练入口（包实现）

职责：依次调用包内未校准训练器，训练 TextCNN、BiLSTM、
Baseline MedBERT 和 Time-aware MedBERT。
输入：已准备好的 Parquet 与词表。
输出：未校准权重、逐折预测和结果表。
状态：历史未校准训练编排器；不执行数据构建、传统 ML、解释或制图，文件名中的
“run_all”仅指四个深度模型实验，不代表完整端到端流水线。
"""

from diliplus.config import load_settings
from diliplus.training.deep_trainer import main as train_deep_model

def run_experiments(settings=None):
    settings = settings or load_settings()
    
    # 按照消融实验的逻辑顺序编排
    models_to_train = [
        'MultiModalTextCNN', 
        'MultiModalBiLSTM', 
        'MultiModalBaselineMedBERT',
        'MultiModalTimeAwareMedBERT'
    ]
    
    print("🚀 Starting Automated Experiment Pipeline...")
    print(f"📋 Target Models: {models_to_train}\n")
    
    for model_name in models_to_train:
        print("="*70)
        print(f"🟢 [LAUNCHING PROCESS] Training Model: {model_name}")
        print("="*70)
        
        # 严格对齐期望的超参数：Batch Size = 256, Epochs = 50, LR = 1e-4
        train_deep_model(["--model", model_name], settings=settings)
        print(f"\n✅ [SUCCESS] Finished training {model_name}.")

if __name__ == "__main__":
    run_experiments()
