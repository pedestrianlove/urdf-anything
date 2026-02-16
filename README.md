# URDF-Anything

Code for training and inference of URDF (joint/link) prediction from images.
**依赖安装顺序**（避免 torch-cluster 等报错）：

1. **先安装 PyTorch**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   ```
   （按本机 CUDA 选 `cu124` / `cu121` / `cpu`。）

2. **再安装本仓库依赖**：
   ```bash
   pip install -r requirements.txt -i https://pypi.org/simple/
   ```

3. **可选：安装 torch-cluster**。必须在已安装 torch 之后执行：
   ```bash
   pip install torch-cluster --no-build-isolation
   ```
   若从源码构建报错 `No module named 'torch'`，是因为 pip 默认的构建隔离环境里没有 torch。可任选其一：
   - **使用无构建隔离**（在当前环境里构建，能用到已装的 torch）：
     ```bash
     pip install torch-cluster --no-build-isolation
     ```
   - **使用 PyG 预编译轮子**（替换为你的 PyTorch 与 CUDA 版本）：
     ```bash
     pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
     ```

4. **可选：安装 diso**（TripoSG 网格提取等会用到）。构建时需要 torch，须在已安装 torch 后执行：
   ```bash
   pip install diso --no-build-isolation
   ```

**Hugging Face 认证**（下载模型前建议先配置）：

1. 在 [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens) 创建 Token（Read 权限即可）。
2. 任选一种方式配置：
   - **命令行登录**（推荐）：`huggingface-cli login`，按提示粘贴 Token。
   - **环境变量**：`export HF_TOKEN=你的token` 或 `export HUGGING_FACE_HUB_TOKEN=你的token`（可写入 `~/.bashrc` 或 `~/.zshrc`）。

**Setup:** Clone [TripoSG](https://github.com/VAST-AI-Research/TripoSG) (used for 3D geometry)，并按 TripoSG 官方结构在 `TripoSG/pretrained_weights/` 下下载权重（参考你在 V2 中的 `TripoSG_src/pretrained_weights`）：

```bash
# 1) Clone TripoSG code（在 codes/ 下）
git clone https://github.com/VAST-AI-Research/TripoSG.git

# 2) 下载 TripoSG 主模型（含 transformer / vae / model_index.json 等）
huggingface-cli download VAST-AI/TripoSG --local-dir TripoSG/pretrained_weights/TripoSG

# 3) 下载 RMBG-1.4 背景移除模型
huggingface-cli download briaai/RMBG-1.4 --local-dir TripoSG/pretrained_weights/RMBG-1.4

# 4) 下载 DINOv3 图像编码器（供 cache 构建与推理使用）
huggingface-cli download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir DINOv3
```

若未安装 `huggingface-cli`，也可以用 Python 一次性下载（在 `codes/` 下）：

```bash
python -c "
from huggingface_hub import snapshot_download

# TripoSG 主模型
snapshot_download(repo_id='VAST-AI/TripoSG', local_dir='TripoSG/pretrained_weights/TripoSG')

# RMBG-1.4
snapshot_download(repo_id='briaai/RMBG-1.4', local_dir='TripoSG/pretrained_weights/RMBG-1.4')

# DINOv3
snapshot_download(repo_id='facebook/dinov3-vith16plus-pretrain-lvd1689m', local_dir='DINOv3')
"
```

使用时将 `--dino_model_path` / `--dino_path` 设为 `DINOv3`（或 `DINOv3` 的绝对路径）。



- **Train:** `python scripts/train.py` — 在缓存数据上训练 URDF 模型；多机多卡见 `scripts/run_multi_node_training.sh`。
- **Inference:** `python scripts/inference.py` — 加载 checkpoint，从输入图像预测 URDF。
- **Cache:** `python scripts/build_cache.py` — 构建训练用缓存数据集（VAE + DINO 编码，生成 train/test 划分）。
- **Normalize:** `normalize.py` — 数据归一化工具。

核心逻辑在包 `urdf_anything` 中：`urdf_anything.data`（cache / dataset / build_cache）、`urdf_anything.model`、`urdf_anything.training`、`urdf_anything.inference`。脚本为薄入口，配置见 `urdf_anything/model/URDFModel_config.yaml`，3D 几何使用 TripoSG。
