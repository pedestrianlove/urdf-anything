#!/usr/bin/env bash

# 2) Download TripoSG main model (contains transformer / vae / model_index.json etc.)
hf download VAST-AI/TripoSG --local-dir TripoSG/pretrained_weights/TripoSG

# 3) Download RMBG-1.4 background removal model
hf download briaai/RMBG-1.4 --local-dir TripoSG/pretrained_weights/RMBG-1.4

# 4) Download DINOv3 image encoder (used for cache building and inference)
hf download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir DINOv3

