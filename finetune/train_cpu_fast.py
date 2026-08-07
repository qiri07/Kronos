"""
Kronos Qlib Finetune 训练脚本 - CPU 快速版本
适用于快速测试和验证
"""
import os
import sys
import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.kronos import Kronos, KronosTokenizer


class SimpleQlibDataset(Dataset):
    """简化的 Qlib 数据集，用于 CPU 训练"""
    
    def __init__(self, data_path, n_samples=1000, seed=100):
        print(f"加载数据: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.n_samples = n_samples
        self.seed = seed
        self.py_rng = __import__('random').Random(seed)
        
        # 添加时间特征
        for symbol in self.data.keys():
            df = self.data[symbol]
            df = df.reset_index()
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df
        
        # 收集所有有效的索引 - 确保至少有91行数据
        self.indices = []
        for symbol in self.data.keys():
            df = self.data[symbol]
            series_len = len(df)
            # 需要至少91行才能构成一个有效样本
            if series_len >= 91:
                # 只取能构成完整窗口的起始位置 (0 到 series_len - 91)
                for start_idx in range(series_len - 90):
                    self.indices.append((symbol, start_idx))
        
        # 随机采样
        self.indices = self.indices[:n_samples]
        print(f"数据集大小: {len(self.indices)}")
    
    def __len__(self):
        return self.n_samples if self.indices else 1
    
    def __getitem__(self, idx):
        if len(self.indices) == 0:
            # 返回空数据
            x = np.zeros((91, 6), dtype=np.float32)
            x_stamp = np.zeros((91, 5), dtype=np.float32)
            return torch.from_numpy(x), torch.from_numpy(x_stamp)
        
        if idx >= len(self.indices):
            idx = self.py_rng.randint(0, len(self.indices) - 1)
        
        symbol, start_idx = self.indices[idx]
        df = self.data[symbol]
        
        # 获取窗口数据: 90 lookback + 1 predict = 91 rows
        end_idx = start_idx + 91
        win_df = df.iloc[start_idx:end_idx]
        
        # 特征列
        feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
        time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        
        # 提取特征
        x = win_df[feature_cols].values.astype(np.float32)
        x_stamp = win_df[time_cols].values.astype(np.float32)
        
        # 确保形状一致 (防止数据不足的情况)
        if x.shape[0] < 91:
            padding = np.zeros((91 - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.vstack([x, padding])
        if x_stamp.shape[0] < 91:
            padding = np.zeros((91 - x_stamp.shape[0], x_stamp.shape[1]), dtype=np.float32)
            x_stamp = np.vstack([x_stamp, padding])
        
        # 归一化 (使用前90行)
        past_x = x[:90]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0) + 1e-5
        x = (x - x_mean) / x_std
        x = np.clip(x, -5.0, 5.0)
        
        return torch.from_numpy(x), torch.from_numpy(x_stamp)


def main():
    """主训练函数"""
    from finetune.config import Config
    config = Config()
    
    # 快速训练配置
    config.batch_size = 64
    config.n_train_iter = 1000   # 每 epoch 1000 个 batch
    config.n_val_iter = 200      # 验证集 200 个 batch
    config.epochs = 5            # 5 个 epoch
    
    print("=" * 60)
    print(f"Kronos Qlib Finetune (CPU - 快速版本)")
    print(f"batch_size={config.batch_size}, epochs={config.epochs}")
    print(f"训练样本: {config.n_train_iter}, 验证样本: {config.n_val_iter}")
    print(f"数据集路径: {config.dataset_path}")
    print(f"保存路径: {config.save_path}")
    print("=" * 60)
    
    # 设备
    device = torch.device("cpu")
    print(f"\n设备: {device}\n")
    
    # 加载预训练模型
    print("加载预训练模型...")
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    predictor = Kronos.from_pretrained(config.pretrained_predictor_path)
    tokenizer.to(device)
    predictor.to(device)
    tokenizer.eval()
    predictor.train()
    print("模型加载完成!\n")
    
    # 加载数据
    print("加载数据...")
    train_dataset = SimpleQlibDataset(
        f"{config.dataset_path}/train_data.pkl",
        config.n_train_iter,
        config.seed
    )
    val_dataset = SimpleQlibDataset(
        f"{config.dataset_path}/val_data.pkl",
        config.n_val_iter,
        config.seed
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    print(f"训练加载器: {len(train_loader)} batches")
    print(f"验证加载器: {len(val_loader)} batches\n")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=config.predictor_learning_rate
    )
    
    # 保存目录
    save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    # 训练循环
    print("开始训练...\n")
    for epoch in range(config.epochs):
        epoch_start = time.time()
        predictor.train()
        total_loss = 0.0
        
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
            
            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{config.epochs} - "
                      f"Batch {i+1}/{len(train_loader)} - "
                      f"Loss: {total_loss/(i+1):.4f}")
        
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
                loss = predictor.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                
                if isinstance(loss, tuple):
                    loss = loss[0]
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{config.epochs} 完成")
        print(f"  训练 Loss: {total_loss/len(train_loader):.4f}")
        print(f"  验证 Loss: {val_loss:.4f} ({epoch_time:.1f}s)")
        print(f"  总时间: {total_time:.1f}s\n")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            predictor.save_pretrained(save_path)
            print(f"  [OK] 保存最佳模型! (Val Loss: {best_val_loss:.4f})\n")
    
    print("=" * 60)
    print(f"训练完成!")
    print(f"最佳验证 Loss: {best_val_loss:.4f}")
    print(f"总时间: {total_time:.1f}s")
    print(f"模型保存在: {save_dir}/best_model")
    print("=" * 60)


if __name__ == "__main__":
    main()
