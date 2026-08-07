"""
回测 Finetune 模型在 ALI09988 上的表现
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


def load_model():
    """加载 finetuned 模型"""
    print("=" * 60)
    print("加载 Finetune 模型")
    print("=" * 60)
    
    tokenizer = KronosTokenizer.from_pretrained("models/Kronos-Tokenizer-base")
    predictor = Kronos.from_pretrained("outputs/models/finetune_predictor_demo/best_model")
    predictor.eval()
    
    pred = KronosPredictor(predictor, tokenizer, device="cpu", max_context=512)
    print(f"✓ 模型加载完成! 参数量: {sum(p.numel() for p in predictor.parameters()):,}\n")
    
    return pred


def load_data():
    """加载数据"""
    data_path = "./data/HK_ali_09988_kline_5min_all.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    print(f"✓ 加载数据: {len(df)} 条记录")
    print(f"  时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}\n")
    
    return df


def run_backtest(pred, df, lookback=90, pred_len=10, test_ratio=0.2):
    """运行回测"""
    print("=" * 60)
    print("开始回测")
    print("=" * 60)
    
    # 分割训练/测试集
    test_size = int(len(df) * test_ratio)
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    print(f"训练集: {len(train_df)} 条")
    print(f"测试集: {len(test_df)} 条\n")
    
    # 准备测试数据 - 只取部分样本以加快回测
    results = []
    total_samples = 0
    total_time = 0
    
    # 使用测试集的前 100 个样本进行回测 (加快回测速度)
    start_idx = max(lookback, len(df) - test_size)
    max_samples = 100  # 限制回测样本数
    
    print(f"回测进度 (最多 {max_samples} 个样本):")
    for i in range(start_idx, min(start_idx + max_samples * pred_len, len(df) - pred_len), pred_len):
        # 准备输入数据
        window_df = df.iloc[i:i+lookback].copy()
        
        # 准备时间戳
        x_timestamp = pd.Series(window_df['timestamps'].values)
        y_timestamp = pd.Series(pd.date_range(
            start=x_timestamp.iloc[-1] + pd.Timedelta(minutes=5),
            periods=pred_len,
            freq='5min'
        ).values)
        
        # 运行预测
        start_time = time.time()
        pred_df = pred.predict(
            df=window_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        predict_time = time.time() - start_time
        total_time += predict_time
        
        # 计算实际收益
        actual_close = df['close'].iloc[i+lookback:i+lookback+pred_len].values
        pred_close = pred_df['close'].values
        
        # 计算预测准确率
        actual_return = (actual_close[1:] - actual_close[:-1]) / actual_close[:-1]
        pred_return = (pred_close[1:] - pred_close[:-1]) / pred_close[:-1]
        
        # 计算方向准确率
        direction_correct = np.sum(np.sign(actual_return) == np.sign(pred_return)) / len(actual_return)
        
        # 计算 MSE
        mse = np.mean((actual_close - pred_close) ** 2)
        
        # 计算 MAE
        mae = np.mean(np.abs(actual_close - pred_close))
        
        results.append({
            'start_idx': i,
            'actual_close': actual_close.tolist(),
            'pred_close': pred_close.tolist(),
            'actual_return': actual_return.tolist(),
            'pred_return': pred_return.tolist(),
            'direction_correct': direction_correct,
            'mse': mse,
            'mae': mae,
            'time': predict_time
        })
        
        total_samples += 1
        
        if total_samples % 10 == 0:
            print(f"  已完成 {total_samples}/{max_samples} 个预测窗口...")
    
    print(f"\n✓ 回测完成! 共 {total_samples} 个预测窗口")
    print(f"  总耗时: {total_time:.2f}s\n")
    
    return results


def analyze_results(results):
    """分析回测结果"""
    print("=" * 60)
    print("回测结果分析")
    print("=" * 60)
    
    # 提取指标
    directions = [r['direction_correct'] for r in results]
    mses = [r['mse'] for r in results]
    maes = [r['mae'] for r in results]
    
    # 计算平均指标
    avg_direction = np.mean(directions)
    avg_mse = np.mean(mses)
    avg_mae = np.mean(maes)
    
    print(f"\n总体指标:")
    print(f"  平均方向准确率: {avg_direction*100:.2f}%")
    print(f"  平均 MSE: {avg_mse:.4f}")
    print(f"  平均 MAE: {avg_mae:.4f}")
    
    # 计算收益率
    all_actual_returns = []
    all_pred_returns = []
    for r in results:
        all_actual_returns.extend(r['actual_return'])
        all_pred_returns.extend(r['pred_return'])
    
    total_actual_return = np.prod([1 + r for r in all_actual_returns]) - 1
    total_pred_return = np.prod([1 + r for r in all_pred_returns]) - 1
    
    print(f"\n累计收益率:")
    print(f"  实际累计收益率: {total_actual_return*100:+.2f}%")
    print(f"  预测累计收益率: {total_pred_return*100:+.2f}%")
    
    # 计算夏普比率 (简化版)
    returns = np.array(all_pred_returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 78)  # 假设每天78个5分钟周期
    
    print(f"\n风险指标:")
    print(f"  夏普比率: {sharpe:.2f}")
    
    # 按时间段分析
    print(f"\n按时间段分析:")
    for i, r in enumerate(results[:5]):
        print(f"  窗口 {i+1}: 方向准确率={r['direction_correct']*100:.1f}%, MAE={r['mae']:.2f}")
    
    for i, r in enumerate(results[-5:]):
        print(f"  窗口 {len(results)-4+i}: 方向准确率={r['direction_correct']*100:.1f}%, MAE={r['mae']:.2f}")
    
    return {
        'avg_direction': avg_direction,
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'total_actual_return': total_actual_return,
        'total_pred_return': total_pred_return,
        'sharpe': sharpe
    }


def save_results(results, metrics, output_path="./outputs/backtest_results"):
    """保存回测结果"""
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细结果
    detailed_results = []
    for r in results:
        detailed_results.append({
            'start_idx': r['start_idx'],
            'actual_close': r['actual_close'],
            'pred_close': r['pred_close'],
            'direction_correct': r['direction_correct'],
            'mse': r['mse'],
            'mae': r['mae']
        })
    
    result_data = {
        'model': './outputs/models/finetune_predictor_demo/best_model',
        'backtest_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'lookback': 90,
        'pred_len': 10,
        'metrics': metrics,
        'detailed_results': detailed_results
    }
    
    json_path = os.path.join(output_path, f"backtest_ali09988_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 回测结果已保存: {json_path}")
    
    return json_path


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("ALI09988 (阿里巴巴) 回测 - Finetune 模型")
    print("=" * 60 + "\n")
    
    # 1. 加载模型
    pred = load_model()
    
    # 2. 加载数据
    df = load_data()
    if df is None:
        print("❌ 无法加载数据，退出")
        return
    
    # 3. 运行回测
    results = run_backtest(pred, df, lookback=90, pred_len=10, test_ratio=0.2)
    
    # 4. 分析结果
    metrics = analyze_results(results)
    
    # 5. 保存结果
    json_path = save_results(results, metrics)
    
    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)
    print(f"\n总结:")
    print(f"  - 方向准确率: {metrics['avg_direction']*100:.2f}%")
    print(f"  - 平均 MAE: {metrics['avg_mae']:.4f} 元")
    print(f"  - 夏普比率: {metrics['sharpe']:.2f}")
    print(f"  - 预测累计收益: {metrics['total_pred_return']*100:+.2f}%")
    print(f"\n结果文件: {json_path}")


if __name__ == "__main__":
    main()
