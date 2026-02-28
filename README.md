# URDF-Anything+: Autoregressive Articulated 3D Models Generation for Physical Simulation
[[Website](https://urdf-anything-plus.github.io/)] [[arXiv]()] [[Dataset](https://huggingface.co/datasets/URDF-Anything-plus/Dataset)]

## Requirements
### Python environment
1. **Create a conda environment**
   ```bash
   conda create -n urdf-anything python=3.10 -y
   conda activate urdf-anything
   ```

2. **Install PyTorch**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 
   ```

3. **Install dependencies**：
   ```bash
   pip install -r requirements.txt -i https://pypi.org/simple/
   ```

4. **Install torch-cluster**。Must be installed after PyTorch:
   ```bash
   pip install torch-cluster --no-build-isolation
   ```
   
5. **Install diso**（TripoSG mesh extraction will use it）. Must be installed after PyTorch:
   ```bash
   pip install diso --no-build-isolation
   ```

### Pretrained Models (TripoSG & DINOv3)

**Hugging Face authentication**（recommended to configure before downloading models）：

<!-- 1. 在 [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens) 创建 Token（Read 权限即可）。
2. 任选一种方式配置：
   - **命令行登录**（推荐）：`huggingface-cli login`，按提示粘贴 Token。
   - **环境变量**：`export HF_TOKEN=你的token` 或 `export HUGGING_FACE_HUB_TOKEN=你的token`（可写入 `~/.bashrc` 或 `~/.zshrc`）。 -->

**Setup:** Clone [TripoSG](https://github.com/VAST-AI-Research/TripoSG) (used for 3D geometry)，and download the weights in `TripoSG/pretrained_weights/`：

```bash
# 1) Clone TripoSG code
git clone https://github.com/VAST-AI-Research/TripoSG.git

# 2) Download TripoSG main model (contains transformer / vae / model_index.json etc.)
huggingface-cli download VAST-AI/TripoSG --local-dir TripoSG/pretrained_weights/TripoSG

# 3) Download RMBG-1.4 background removal model
huggingface-cli download briaai/RMBG-1.4 --local-dir TripoSG/pretrained_weights/RMBG-1.4

# 4) Download DINOv3 image encoder (used for cache building and inference)
huggingface-cli download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir DINOv3
```

If `huggingface-cli` is not installed, you can also download the models using Python:

```bash
python -c "
from huggingface_hub import snapshot_download

# TripoSG 
snapshot_download(repo_id='VAST-AI/TripoSG', local_dir='TripoSG/pretrained_weights/TripoSG')

# RMBG-1.4
snapshot_download(repo_id='briaai/RMBG-1.4', local_dir='TripoSG/pretrained_weights/RMBG-1.4')

# DINOv3
snapshot_download(repo_id='facebook/dinov3-vith16plus-pretrain-lvd1689m', local_dir='DINOv3')
"
```

## Training
### Data Preparation
Download the dataset from [Hugging Face](https://huggingface.co/datasets/zhuangzhe1229/test_dataset) and unzip it to `data_normalized/`.

The structure of the dataset is as follows:
```
URDF-Anything+ dataset:
├── data_normalized/
│   ├── Laptop_urdf/
│   │   ├── images/
│   │   ├── xxx.obj
│   │   ├── test.urdf/
│   │   ├── info.json/
│   ├── Refrigerator_urdf/
│   │   ├── images/
│   │   ├── xxx.obj
│   │   ├── test.urdf/
│   │   ├── info.json/
│   ├── ...
```

Then run the following command to build the cache:
```bash
python scripts/build_cache.py
```

### Training
```bash
bash scripts/run_multi_node_training.sh
```
You can adjust the training parameters in `scripts/run_multi_node_training.sh`.


