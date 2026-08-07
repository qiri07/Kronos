# Kronos WebUI 使用说明

## 问题修复

**问题**: WebUI 的 "Select Data File" 下拉框为空，无法选择数据文件

**原因**: 
1. WebUI 代码只扫描 `data/` 目录，但项目根目录没有这个目录
2. 没有数据文件时前端没有友好提示

**修复**:
1. 改进 `load_data_files()` 函数，扫描多个候选目录
2. 当没有数据文件时返回友好的提示信息
3. 创建 `data/` 目录并复制示例数据文件
4. 更新前端代码以正确处理空状态

## 快速开始

### 1. 启动 WebUI

```bash
cd webui
python run.py
```

或者使用启动脚本:
```bash
cd webui
./start.sh
```

### 2. 访问界面

浏览器打开: http://localhost:7070

### 3. 选择数据文件

在 "Select Data File" 下拉框中应该能看到以下数据文件:

- **HK_ali_09988_kline_5min_all.csv** (5.6 MB) - 港股 5分钟K线数据
- **regression_input.csv** (144 KB) - 回归测试数据

### 4. 加载模型

1. 选择模型: Kronos-base (102.3M 参数)
2. 选择设备: CPU 或 CUDA (如果有 NVIDIA GPU)
3. 点击 "Load Model" 按钮

### 5. 加载数据

1. 选择数据文件
2. 点击 "Load Data" 按钮
3. 查看数据信息（行数、时间范围、价格范围等）

### 6. 预测

1. 调整预测参数（温度、top_p、采样次数）
2. 拖动时间窗口滑块选择预测范围
3. 点击 "Start Prediction" 按钮
4. 查看预测结果图表和对比分析

## 数据文件格式要求

CSV 文件需要包含以下列:

**必需列**:
- `timestamps` 或 `date`: 时间戳
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价

**可选列**:
- `volume`: 成交量
- `amount`: 成交额

## 常见问题

### Q: 仍然看不到数据文件?

**A**: 检查以下几点:
1. 确认 `data/` 目录下有 CSV 文件
2. 检查控制台输出是否显示 "Found X data file(s)"
3. 确认 CSV 文件包含必需列

### Q: 如何添加自己的数据文件?

**A**: 将 CSV 文件复制到 `data/` 目录:
```bash
cp your_data.csv data/
```

文件需要满足上述格式要求。

### Q: 模型加载失败?

**A**: 确保已下载模型:
```bash
python download_models.py
```

### Q: 数据文件格式不对?

**A**: 使用以下命令检查数据格式:
```python
import pandas as pd
df = pd.read_csv('data/your_file.csv')
print(df.head())
print(df.columns)
```

## 配置文件位置

| 文件 | 路径 | 用途 |
|------|------|------|
| 模型路径 | `finetune/config.py` | 微调配置 |
| 数据目录 | `data/` | WebUI 数据文件 |
| 模型目录 | `models/` | 预训练模型 |

## 技术支持

如有问题，请检查:
1. 控制台输出的错误信息
2. 浏览器开发者工具的 Network 标签
3. `webui/prediction_results/` 目录下的预测日志
