"""
Kronos CLI - Minimal prediction script
Usage: python predict.py --data <csv_file> --lookback 400 --pred_len 120
"""
import argparse
import pandas as pd
import torch
from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    parser = argparse.ArgumentParser(description='Kronos Financial Prediction CLI')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--lookback', type=int, default=400, help='Historical context length')
    parser.add_argument('--pred_len', type=int, default=120, help='Prediction length')
    parser.add_argument('--model', type=str, default='NeoQuasar/Kronos-small', help='Model name')
    parser.add_argument('--tokenizer', type=str, default='NeoQuasar/Kronos-Tokenizer-base', help='Tokenizer name')
    parser.add_argument('--output', type=str, help='Output CSV path')
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    x_df = df.loc[:args.lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:args.lookback-1, 'timestamps']
    y_timestamp = df.loc[args.lookback:args.lookback+args.pred_len-1, 'timestamps']

    print("Predicting...")
    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=args.pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

    print("\nPrediction results:")
    print(pred_df.head())

    if args.output:
        pred_df.to_csv(args.output)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
