"""
Kronos Qlib Finetune 训练脚本 - CPU 优化版本 (中等规模)
适用于没有 GPU 或 GPU 不兼容的情况
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


class SimpleQlibDataset(Dataset):
    """简化的 Qlib 数据集，用于 CPU 训练"""
    
    def __init__(self, data_path, n_samples=10000, seed=100):
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
        
        # 收集所有有效的索引
        self.indices = []
        for symbol in self.data.keys():
            df = self.data[symbol]
            series_len = len(df)
            if series_len >= 101:  # 需要至少 90 + 10 + 1
                for start_idx in range(series_len - 10):
                    self.indices.append((symbol, start_idx))
        
        # 随机采样
        self.indices = self.indices[:n_samples]
        print(f"数据集大小: {len(self.indices)}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if idx >= len(self.indices):
            idx = self.py_rng.randint(0, len(self.indices) - 1)
        
        symbol, start_idx = self.indices[idx]
        df = self.data[symbol]
        
        # 获取窗口数据
        end_idx = min(start_idx + 101, len(df))
        win_df = df.iloc[start_idx:end_idx]
        
        # 特征列
        feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
        time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        
        # 确保有足够的数据
        if len(win_df) < 91:
            # 填充到 91 个时间点
            padding = pd.DataFrame({
                col: [0.0] * (91 - len(win_df))
                for col in win_df.columns
            })
            win_df = pd.concat([win_df, padding], ignore_index=True)
            win_df = win_df.iloc[:91]
        
        # 提取特征
        x = win_df[feature_cols].values.astype(np.float32)
        x_stamp = win_df[time_cols].values.astype(np.float32)
        
        # 归一化
        if len(x) >= 2:
            past_x = x[:-1]
            mean = np.mean(past_x, axis=0)
            std = np.std(past_x, axis=0) + 1e-5
            x = (x - mean) / std
            x = np.clip(x, -5.0, 5.0)
        
        return torch.from_numpy(x), torch.from_numpy(x_stamp)


def main():
    """主训练函数"""
    from finetune.config import Config
    config = Config()
    
    # 修改为中等规模训练（适合 CPU）
    config.batch_size = 32
    config.n_train_iter = 10000  # 每 epoch 10000 个 batch
    config.n_val_iter = 2000     # 验证集 2000 个 batch
    config.epochs = 10           # 10 个 epoch
    
    print("=" * 60)
    print(f"Kronos Qlib Finetune (CPU - 中等规模)")
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
        num_workers=0  # CPU 训练，不使用多线程避免内存问题
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
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
            print(f"  保存最佳模型! (Val Loss: {best_val_loss:.4f})\n")
    
    print("=" * 60)
    print(f"训练完成!")
    print(f"最佳验证 Loss: {best_val_loss:.4f}")
    print(f"总时间: {total_time:.1f}s")
    print(f"模型保存在: {save_dir}/best_model")
    print("=" * 60)


if __name__ == "__main__":
    main()
