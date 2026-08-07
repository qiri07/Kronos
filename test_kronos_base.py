"""Test script to verify downloaded Kronos-base model works correctly."""
import numpy as np
import pandas as pd
import torch
import sys

sys.path.append(".")
from model import Kronos, KronosTokenizer, KronosPredictor

# 配置
DEVICE = "cpu"
MAX_CTX_LEN = 512
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]
TEST_DATA_PATH = "./tests/data/regression_input.csv"
OUTPUT_PATH = "./tests/data/regression_output_base_512.csv"

print("=" * 60)
print("Testing Kronos-base model")
print("=" * 60)

# 1. 加载模型
print("\n1. Loading model and tokenizer...")
tokenizer = KronosTokenizer.from_pretrained("./models/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("./models/Kronos-base")
tokenizer.eval()
model.eval()

predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)
print("   ✓ Model loaded successfully!")

# 2. 加载测试数据
print("\n2. Loading test data...")
df = pd.read_csv(TEST_DATA_PATH, parse_dates=["timestamps"])
print(f"   Total data points: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# 3. 生成预测
print("\n3. Making predictions...")
context_len = 512
pred_len = 8

context_df = df.iloc[:context_len].copy()
context_features = context_df[FEATURE_NAMES].reset_index(drop=True)
x_timestamp = context_df["timestamps"].reset_index(drop=True)
future_timestamp = df["timestamps"].iloc[context_len:context_len + pred_len].reset_index(drop=True)

with torch.no_grad():
    pred_df = predictor.predict(
        df=context_features,
        x_timestamp=x_timestamp,
        y_timestamp=future_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_k=1,
        top_p=1.0,
        verbose=True,
        sample_count=1,
    )

print("\n4. Results:")
print(f"   Predicted shape: {pred_df.shape}")
print(f"   Columns: {list(pred_df.columns)}")
print("\n   First 3 predictions:")
print(pred_df.head(3).round(4))

# 5. 保存结果
pred_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n5. ✓ Predictions saved to: {OUTPUT_PATH}")

# 6. 对比已知的 small 模型输出，确认差异
print("\n" + "=" * 60)
print("Comparison with Kronos-small:")
print("=" * 60)

# 加载 small 模型的预期输出
expected_small = pd.read_csv("./tests/data/regression_output_512.csv", parse_dates=["timestamps"])
print(f"   Kronos-small MSE: {np.mean((pred_df[FEATURE_NAMES].values - expected_small[FEATURE_NAMES].values)**2):.6f}")

print("\n✅ Test completed successfully!")
