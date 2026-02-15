"""
Thin cache-build entry: run urdf_anything.data.build_cache.main().
Run from repo root: python scripts/build_cache.py [--token_length 512] [--cache_dir cache] [--dino_model_path DINOv3]  ...
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from urdf_anything.data.build_cache import main

if __name__ == "__main__":
    main()
