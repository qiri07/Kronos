"""
使用 Finetune 模型预测阿里巴巴 (ALI09988)
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.kronos import Kronos, KronosTokenizer, KronosPredictor


def load_models():
    """加载 finetuned 模型"""
    print("=" * 60)
    print("加载 Finetune 模型")
    print("=" * 60)
    
    # 加载 tokenizer
    tokenizer = KronosTokenizer.from_pretrained("models/Kronos-Tokenizer-base")
    print("✓ Tokenizer 加载完成")
    
    # 加载 predictor
    predictor = Kronos.from_pretrained("outputs/models/finetune_predictor_demo/best_model")
    predictor.eval()
    print("✓ Predictor 加载完成")
    print(f"  参数量: {sum(p.numel() for p in predictor.parameters()):,}")
    
    # 创建预测器
    pred = KronosPredictor(predictor, tokenizer, device="cpu", max_context=512)
    print("✓ Predictor 实例化完成\n")
    
    return pred


def load_stock_data(stock_code="09988"):
    """加载阿里巴巴股票数据"""
    data_path = "./data/HK_ali_09988_kline_5min_all.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✓ 加载数据: {len(df)} 条记录")
    print(f"  时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    print(f"  最新收盘价: {df['close'].iloc[-1]:.2f} 元\n")
    
    # 确保时间戳格式正确
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    return df


def prepare_prediction_data(df, lookback=90, pred_len=10):
    """准备预测数据"""
    # 选择特征列
    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
    
    # 确保有足够的数据
    if len(df) < lookback:
        print(f"❌ 数据不足: 需要 {lookback} 条，只有 {len(df)} 条")
        return None
    
    # 取最后 lookback 条作为输入
    history_df = df.iloc[-lookback:].copy()
    
    # 准备时间戳 (必须是 pandas Series)
    x_timestamp = pd.Series(history_df['timestamps'].values)
    y_timestamp = pd.Series(pd.date_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(minutes=5),
        periods=pred_len,
        freq='5min'
    ).values)
    
    return history_df, x_timestamp, y_timestamp


def run_prediction(pred, history_df, x_timestamp, y_timestamp, pred_len=10):
    """运行预测"""
    print("=" * 60)
    print("开始预测")
    print("=" * 60)
    
    start_time = time.time()
    
    # 调用预测
    pred_df = pred.predict(
        df=history_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )
    
    predict_time = time.time() - start_time
    print(f"\n✓ 预测完成! 耗时: {predict_time:.2f}s\n")
    
    return pred_df


def analyze_prediction(pred_df, history_df):
    """分析预测结果"""
    print("=" * 60)
    print("预测结果分析")
    print("=" * 60)
    
    # 获取最新价格
    last_close = history_df['close'].iloc[-1]
    
    # 提取预测收盘价
    pred_close = pred_df['close'].values
    
    print(f"\n当前收盘价: {last_close:.2f} 元")
    print(f"\n预测未来 {len(pred_close)} 个时间点收盘价:")
    print("-" * 50)
    
    total_change = 0
    for i, price in enumerate(pred_close):
        change = (price - last_close) / last_close * 100
        total_change += change
        print(f"  Step {i+1}: {price:.2f} 元 ({'+' if change >= 0 else ''}{change:.2f}%)")
    
    avg_change = total_change / len(pred_close)
    print("-" * 50)
    print(f"  平均涨跌幅: {avg_change:+.2f}%")
    print(f"  预测方向: {'看涨' if avg_change > 0 else '看跌'}")
    
    return pred_close


def save_results(pred_df, history_df, stock_code, output_dir="./webui/prediction_results"):
    """保存预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存 JSON
    result = {
        'stock_code': stock_code,
        'stock_name': '阿里巴巴',
        'model_path': './outputs/models/finetune_predictor_demo/best_model',
        'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'lookback': len(history_df),
        'pred_len': len(pred_df),
        'last_close': float(history_df['close'].iloc[-1]),
        'predictions': pred_df['close'].tolist(),
        'timestamps': [str(t) for t in pred_df.index]
    }
    
    json_path = os.path.join(output_dir, f"prediction_{stock_code}_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 预测结果已保存: {json_path}")
    
    return json_path


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("ALI09988 (阿里巴巴) 股票预测 - Finetune 模型")
    print("=" * 60 + "\n")
    
    # 1. 加载模型
    pred = load_models()
    
    # 2. 加载数据
    df = load_stock_data("09988")
    if df is None:
        print("❌ 无法加载数据，退出")
        return
    
    # 3. 准备数据
    lookback = 90
    pred_len = 10
    history_df, x_timestamp, y_timestamp = prepare_prediction_data(df, lookback, pred_len)
    
    if history_df is None:
        print("❌ 数据准备失败，退出")
        return
    
    # 4. 运行预测
    pred_df = run_prediction(pred, history_df, x_timestamp, y_timestamp, pred_len)
    
    # 5. 分析结果
    pred_close = analyze_prediction(pred_df, history_df)
    
    # 6. 保存结果
    json_path = save_results(pred_df, history_df, "09988")
    
    print("\n" + "=" * 60)
    print("预测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
