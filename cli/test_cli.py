"""
CLI 最小版本测试脚本
测试内容:
1. 模型加载测试
2. 预测功能测试
3. 输出结果验证
"""
import sys
import os
import pandas as pd
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from cli.model import Kronos, KronosTokenizer, KronosPredictor

# 测试数据
TEST_DATA_PATH = Path(__file__).parent.parent / "data" / "regression_input.csv"
OUTPUT_PATH = Path(__file__).parent / "test_output.csv"

def test_model_loading():
    """测试1: 模型加载"""
    print("=" * 50)
    print("测试1: 模型加载...")
    try:
        print("  - 加载 Tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        print("  ✓ Tokenizer 加载成功")

        print("  - 加载 Model...")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("  ✓ Model 加载成功")

        print("  - 创建 Predictor...")
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        print("  ✓ Predictor 创建成功")

        return predictor, tokenizer, model
    except Exception as e:
        print(f" ✗ 模型加载失败: {e}")
        return None, None, None

def test_prediction(predictor):
    """测试2: 预测功能"""
    print("\n" + "=" * 50)
    print("测试2: 预测功能...")

    try:
        # 加载测试数据
        print("  - 加载测试数据...")
        df = pd.read_csv(TEST_DATA_PATH, parse_dates=["timestamps"])
        print(f"  ✓ 数据加载成功, 共 {len(df)} 行")

        # 准备数据
        lookback = 400
        pred_len = 50

        x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.loc[:lookback-1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

        print(f"  - 历史数据: {len(x_df)} 行")
        print(f"  - 预测长度: {pred_len} 行")

        # 执行预测
        print("  - 执行预测...")
        with torch.no_grad():
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=True
            )

        print(f"  ✓ 预测成功, 输出形状: {pred_df.shape}")
        print(f"  ✓ 输出列: {list(pred_df.columns)}")

        # 保存结果
        pred_df.to_csv(OUTPUT_PATH)
        print(f"  ✓ 结果已保存到: {OUTPUT_PATH}")

        return pred_df
    except Exception as e:
        print(f" ✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_output_validity(pred_df):
    """测试3: 输出结果验证"""
    print("\n" + "=" * 50)
    print("测试3: 输出结果验证...")

    if pred_df is None:
        print(" ✗ 没有预测结果")
        return False

    # 验证列名
    expected_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    actual_columns = list(pred_df.columns)

    print(f"  - 检查列名...")
    if set(expected_columns).issubset(set(actual_columns)):
        print(f"  ✓ 列名验证通过: {actual_columns}")
    else:
        print(f"  ✗ 列名不匹配! 期望: {expected_columns}, 实际: {actual_columns}")
        return False

    # 验证数据有效性
    print(f"  - 检查数据有效性...")
    numeric_cols = ['open', 'high', 'low', 'close']
    all_valid = True

    for col in numeric_cols:
        if col in pred_df.columns:
            non_null = pred_df[col].notna().sum()
            if non_null == len(pred_df):
                print(f"  ✓ {col}: 全部有效 ({len(pred_df)} 个值)")
            else:
                print(f"  ⚠ {col}: 有 {len(pred_df) - non_null} 个缺失值")
                all_valid = False

    # 检查数值范围
    print(f"  - 检查数值范围...")
    for col in numeric_cols:
        if col in pred_df.columns:
            min_val = pred_df[col].min()
            max_val = pred_df[col].max()
            if min_val >= 0 and max_val > min_val:
                print(f"  ✓ {col}: [{min_val:.2f}, {max_val:.2f}]")
            else:
                print(f"  ✗ {col}: 数值异常 [{min_val}, {max_val}]")
                all_valid = False

    return all_valid

def test_cli_script():
    """测试4: CLI 脚本导入测试"""
    print("\n" + "=" * 50)
    print("测试4: CLI 脚本导入测试...")

    try:
        # 模拟命令行参数
        sys.argv = ['predict.py', '--data', str(TEST_DATA_PATH), '--lookback', '400', '--pred_len', '50']
        print("  - 检查 CLI 脚本语法...")

        # 读取脚本并检查
        cli_script = Path(__file__).parent / "predict.py"
        with open(cli_script) as f:
            code = f.read()

        compile(code, cli_script, 'exec')
        print("  ✓ CLI 脚本语法正确")
        print(f"  ✓ 脚本路径: {cli_script}")

        return True
    except Exception as e:
        print(f" ✗ CLI 脚本测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("\n" + "=" * 50)
    print("Kronos CLI 最小版本测试")
    print("=" * 50)

    results = {
        'model_loading': False,
        'prediction': False,
        'output_validity': False,
        'cli_script': False
    }

    # 测试1: 模型加载
    predictor, tokenizer, model = test_model_loading()
    if predictor:
        results['model_loading'] = True

    # 测试2: 预测功能
    if predictor:
        pred_df = test_prediction(predictor)
        if pred_df is not None:
            results['prediction'] = True

        # 测试3: 输出验证
        if pred_df is not None:
            results['output_validity'] = test_output_validity(pred_df)
    else:
        print("\n⚠ 跳过预测相关测试 (模型加载失败)")

    # 测试4: CLI 脚本测试
    results['cli_script'] = test_cli_script()

    # 汇总结果
    print("\n" + "=" * 50)
    print("测试汇总")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")

    print(f"\n总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("\n🎉 所有测试通过! CLI 最小版本功能正常。")
        return 0
    else:
        print("\n⚠ 部分测试失败，请检查上述错误信息。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
