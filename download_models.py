"""Download Kronos models from Hugging Face Hub."""
from huggingface_hub import snapshot_download

# 选择你需要下载的模型组合
# 组合1: Kronos-mini (最小，4.1M 参数)
# snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="./models/Kronos-Tokenizer-2k")
# snapshot_download(repo_id="NeoQuasar/Kronos-mini",      local_dir="./models/Kronos-mini")

# 组合2: Kronos-small (24.7M 参数，推荐入门)
# snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="./models/Kronos-Tokenizer-base")
# snapshot_download(repo_id="NeoQuasar/Kronos-small",          local_dir="./models/Kronos-small")

# 组合3: Kronos-base (102.3M 参数，效果更好)
snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="./models/Kronos-Tokenizer-base")
snapshot_download(repo_id="NeoQuasar/Kronos-base",           local_dir="./models/Kronos-base")

print("下载完成！模型保存在 ./models/ 目录下。")
