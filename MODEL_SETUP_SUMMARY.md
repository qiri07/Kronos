# Kronos 模型下载与配置总结

## 已完成的操作

### 1. 模型下载 ✅
- **Kronos-Tokenizer-base**: 词表模型，将 K线数据编码为 token
- **Kronos-base**: 102.3M 参数的预测模型

模型保存位置：
```
models/
├── Kronos-base/
│   ├── config.json
│   ├── model.safetensors
│   └── README.md
└── Kronos-Tokenizer-base/
    ├── config.json
    ├── model.safetensors
    └── README.md
```

### 2. 配置文件更新 ✅
修改了 `finetune/config.py` 中的模型路径：
```python
self.pretrained_tokenizer_path = "./models/Kronos-Tokenizer-base"
self.pretrained_predictor_path = "./models/Kronos-base"
```

### 3. 测试验证 ✅
- 自定义测试脚本 `test_kronos_base.py` 运行成功
- 项目回归测试 4/4 通过

## 后续使用指南

### 方式一：直接预测
```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("./models/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("./models/Kronos-base")
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

### 方式二：微调（Finetune）
1. 准备 Qlib 数据
2. 运行数据预处理：
   ```bash
   python finetune/qlib_data_preprocess.py
   ```
3. 微调词表：
   ```bash
   torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
   ```
4. 微调预测器：
   ```bash
   torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
   ```
5. 回测评估：
   ```bash
   python finetune/qlib_test.py --device cuda:0
   ```

### 方式三：WebUI 界面
```bash
cd webui
python app.py
```

## 注意事项

1. **max_context**: Kronos-base 的最大上下文长度为 512
2. **输入数据**: 需要包含 `open`, `high`, `low`, `close` 列（`volume` 和 `amount` 可选）
3. **设备支持**: 支持 CPU 和 GPU（CUDA）

## 下载脚本位置
- `download_models.py`: 用于下载更多模型变体（mini/small）
- `test_kronos_base.py`: 快速验证模型能否正常工作
