"""
Kronos Qlib Finetune 训练脚本 (CPU 优化版本)
减少数据量，加快训练速度
"""
import os
import sys
import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.kronos import Kronos, KronosTokenizer
from config import Config


class SimpleQlibDataset(Dataset):
    def __init__(self, data_path, n_samples=500, seed=100):
        print(f"加载数据: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.n_samples = n_samples
        self.seed = seed
        self.py_rng = __import__('random').Random(seed)
        
        # 添加时间特征
        for symbol, df in self.data.items():
            df = df.reset_index()
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df
        
        self.indices = []
        for symbol, df in self.data.items():
            series_len = len(df)
            window = 90 + 10 + 1
            if series_len >= window:
                self.indices.extend([(symbol, i) for i in range(series_len - window + 1)])
        
        self.indices = self.indices[:n_samples]
        print(f"数据集大小: {len(self.indices)}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if idx >= len(self.indices):
            idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[idx]
        df = self.data[symbol]
        end_idx = start_idx + 90 + 10 + 1
        win_df = df.iloc[start_idx:end_idx]
        
        feature_cols = [c for c in ['open', 'high', 'low', 'close', 'vol', 'amt'] if c in win_df.columns]
        time_cols = [c for c in ['minute', 'hour', 'weekday', 'day', 'month'] if c in win_df.columns]
        
        x = win_df[feature_cols].values.astype(np.float32)
        x_stamp = win_df[time_cols].values.astype(np.float32) if time_cols else np.zeros((91, 5), dtype=np.float32)
        
        past_x = x[:90]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0) + 1e-5
        x = (x - x_mean) / x_std
        x = np.clip(x, -5.0, 5.0)
        
        return torch.from_numpy(x), torch.from_numpy(x_stamp)


def main():
    print("=" * 60, flush=True)
    print("Kronos Qlib Finetune (CPU 优化版)", flush=True)
    print("=" * 60, flush=True)
    
    config = Config()
    config.use_comet = False
    config.batch_size = 16  # 增大 batch size
    config.n_train_iter = 500  # 减少训练样本数
    config.n_val_iter = 100   # 减少验证样本数
    
    print(f"batch_size={config.batch_size}, epochs={config.epochs}", flush=True)
    print(f"训练样本: {config.n_train_iter}, 验证样本: {config.n_val_iter}", flush=True)
    print(f"数据集路径: {config.dataset_path}", flush=True)
    print(f"保存路径: {config.save_path}", flush=True)
    
    device = torch.device("cpu")
    print(f"设备: {device}", flush=True)
    
    print("\n加载预训练模型...", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    predictor = Kronos.from_pretrained(config.pretrained_predictor_path)
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.train()
    
    print("\n加载数据...", flush=True)
    train_dataset = SimpleQlibDataset(f"{config.dataset_path}/train_data.pkl", config.n_train_iter, config.seed)
    val_dataset = SimpleQlibDataset(f"{config.dataset_path}/val_data.pkl", config.n_val_iter, config.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    print(f"训练集: {len(train_loader)} batches, 验证集: {len(val_loader)} batches", flush=True)
    
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.predictor_learning_rate)
    save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n开始训练...", flush=True)
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        predictor.train()
        total_loss = 0.0
        n_batches = 0
        
        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            
            with torch.no_grad():
                token_indices = tokenizer.encode(batch_x, half=True)
                token_in = [token_indices[0][:, :-1], token_indices[1][:, :-1]]
                token_out = [token_indices[0][:, 1:], token_indices[1][:, 1:]]
            
            logits = predictor(token_in[0], token_in[1], batch_x_stamp[:, :-1])
            loss = predictor.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            if isinstance(loss, tuple):
                loss = loss[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=3.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} - Batch {i+1}/{len(train_loader)} - Loss: {total_loss/(i+1):.4f}", flush=True)
        
        avg_loss = total_loss / n_batches
        
        # 验证
        predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device)
                batch_x_stamp = batch_x_stamp.to(device)
                
                token_indices = tokenizer.encode(batch_x, half=True)
                token_in = [token_indices[0][:, :-1], token_indices[1][:, :-1]]
                token_out = [token_indices[0][:, 1:], token_indices[1][:, 1:]]
                
                logits = predictor(token_in[0], token_in[1], batch_x_stamp[:, :-1])
                vl = predictor.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                if isinstance(vl, tuple):
                    vl = vl[0]
                val_loss += vl.item()
        
        val_loss /= len(val_loader)
        elapsed = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train: {avg_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "checkpoints", "best_model")
            os.makedirs(save_path, exist_ok=True)
            predictor.save_pretrained(save_path)
            print(f"  -> 保存最佳模型 (Val Loss: {best_val_loss:.4f})", flush=True)
    
    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time:.1f}s", flush=True)
    print(f"最佳验证损失: {best_val_loss:.4f}", flush=True)
    print(f"模型保存路径: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
