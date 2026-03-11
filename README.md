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
   pip install -r  -u requirements.txt -i https://pypi.org/simple/
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

Important: There is a little problem with TripoSG/triposg/models/autoencoders/autoencoder_kl_triposg.py, you need to uncomment the line `from torch_cluster import fps`.

## Training
### Data Preparation
Download the dataset from [Hugging Face](https://huggingface.co/datasets/URDF-Anything-plus/Dataset) and unzip it to `data_normalized/`.

The structure of the dataset is as follows:
```
URDF-Anything-plus:
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
bash scripts/run_multi_node_training.sh [node_rank] [master_addr] [nproc_per_node] [training parameters...]
```
For example, to train on 1 machine with 8 GPUs, you can run:
```bash
bash scripts/run_multi_node_training.sh 0 localhost 8
```
You can adjust the training parameters in `scripts/run_multi_node_training.sh`.

In pretraining stage, we use the following hyperparameters:
```yaml
--init_mode train_from_scratch
```
In finetuning stage, we use the following hyperparameters:
```yaml
--init_mode resume_from_ckpt
--checkpoint_path CHECKPOINT_PATH FROM PRETRAINING STAGE
--train_urdf_params True
--train_eot True
```

## Inference
```bash
bash scripts/inference.sh
```
If you are in 'in_the_wild' mode, you need to check the orientation of the generated mesh:
```bash
可选：在终端中对 mesh 做简单旋转。可多次旋转，直接回车结束。
  0: 不旋转（直接回车也等价于结束）
  1-6: 围绕 X/Y/Z 轴 ±90°
angle_map_print: {'1': '绕X轴旋转90度', '2': '绕X轴旋转-90度', '3': '绕Y轴旋转90度', '4': '绕Y轴旋转-90度', '5': '绕Z轴旋转90度', '6': '绕Z轴旋转-90度'}
输入旋转编号并回车（直接回车=确认并结束旋转）[0-6]: 
```
You need to make sure the object is oriented towards the positive z direction.