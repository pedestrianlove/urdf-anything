import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch

# 确保 TripoSG 代码可以同时以 `TripoSG.triposg` 和 `triposg` 方式导入（和 inference.py 一致）
_CODES_DIR = os.path.dirname(os.path.abspath(__file__))
_TRIPOSG_ROOT = os.path.join(_CODES_DIR, "TripoSG")
if _TRIPOSG_ROOT not in sys.path:
    sys.path.insert(0, _TRIPOSG_ROOT)

from urdf_anything.inference import URDFInference
from urdf_anything.data.cache import load_cache_data


@dataclass
class InferenceTask:
    image_path: str
    whole_mesh_path: str
    output_dir: str
    cache_name: str
    obj_id: str
    whole_image_name: str


def discover_cache_paths(cache_root: str) -> List[str]:
    """
    在 cache_root 下查找所有包含 metadata.json 和 test_data.pt 的子目录。
    例如：cache-dinov3-h-normalize/laptop_eot_token512 等。
    """
    cache_paths: List[str] = []
    for name in os.listdir(cache_root):
        subdir = os.path.join(cache_root, name)
        if not os.path.isdir(subdir):
            continue
        meta_path = os.path.join(subdir, "metadata.json")
        test_path = os.path.join(subdir, "test_data.pt")
        if os.path.exists(meta_path) and os.path.exists(test_path):
            cache_paths.append(subdir)
    return cache_paths


def collect_tasks_from_cache(cache_path: str) -> Tuple[List[InferenceTask], Dict]:
    """
    从单个 cache 目录（如 cache-dinov3-h-normalize/laptop_eot_token512）中，
    读取 test split，并按照 (obj_id, whole_image_name) 去重生成 InferenceTask。
    """
    data_index, metadata = load_cache_data(cache_path, split="test")
    data_root = metadata["data_root"]
    cache_name = metadata.get("cache_name", os.path.basename(cache_path))

    # 去重：同一个 obj_id + whole_image_name 只推理一次
    seen_keys = set()
    tasks: List[InferenceTask] = []

    for entry in data_index:
        obj_id = entry["obj_id"]
        whole_image_name = entry["whole_image_name"]
        key = (obj_id, whole_image_name)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        image_path = os.path.join(
            data_root,
            obj_id,
            "images",
            whole_image_name,
        )
        whole_mesh_path = os.path.join(
            data_root,
            obj_id,
            "whole.obj",
        )
        # 输出目录：output_batch/<cache_name>/<obj_id>/<view_name>
        view_name = os.path.splitext(whole_image_name)[0]
        output_dir = os.path.join(
            _CODES_DIR,
            "output_batch",
            cache_name,
            obj_id,
            view_name,
        )

        tasks.append(
            InferenceTask(
                image_path=image_path,
                whole_mesh_path=whole_mesh_path,
                output_dir=output_dir,
                cache_name=cache_name,
                obj_id=obj_id,
                whole_image_name=whole_image_name,
            )
        )

    return tasks, metadata


def split_tasks_among_devices(
    tasks: List[InferenceTask],
    devices: List[str],
) -> Dict[str, List[InferenceTask]]:
    """
    简单轮询，把任务按设备均匀分配。
    """
    assignment: Dict[str, List[InferenceTask]] = {d: [] for d in devices}
    for idx, task in enumerate(tasks):
        dev = devices[idx % len(devices)]
        assignment[dev].append(task)
    return assignment


def worker(
    device: str,
    tasks: List[InferenceTask],
    model_path: str,
    model_config_path: str,
    dino_path: str,
    num_tokens: int,
    eot_threshold: float,
    max_reconstruction_attempts: int,
    overlap_chamfer_threshold: float,
) -> None:
    """
    单 GPU 进程：在指定 device 上顺序跑一批任务。
    """
    if not tasks:
        print(f"[{device}] no tasks assigned, exiting.")
        return

    print(f"[{device}] initializing URDFInference, {len(tasks)} tasks.")
    torch.cuda.set_device(int(device.split(":")[-1]))

    inference = URDFInference(
        model_path=model_path,
        model_config_path=model_config_path,
        dino_path=dino_path,
        device=device,
        num_tokens=num_tokens,
        eot_threshold=eot_threshold,
        max_reconstruction_attempts=max_reconstruction_attempts,
        overlap_chamfer_threshold=overlap_chamfer_threshold,
    )

    for t in tasks:
        try:
            os.makedirs(t.output_dir, exist_ok=True)
            print(
                f"[{device}] cache={t.cache_name}, obj={t.obj_id}, "
                f"view={t.whole_image_name}, output={t.output_dir}"
            )
            result = inference.infer_from_image(
                image_path=t.image_path,
                whole_mesh=None,
                whole_mesh_path=t.whole_mesh_path,
                output_dir=t.output_dir,
                save_meshes=True,
                save_urdf=True,
            )
            print(
                f"[{device}] done: obj={t.obj_id}, view={t.whole_image_name}, "
                f"links={result['num_links']}"
            )
        except Exception as e:
            print(
                f"[{device}] ERROR in obj={t.obj_id}, view={t.whole_image_name}: {e}"
            )


def parse_devices_arg(devices_str: str | None) -> List[str]:
    if devices_str:
        ids = [s.strip() for s in devices_str.split(",") if s.strip() != ""]
    else:
        # 默认：如果设置了 CUDA_VISIBLE_DEVICES，则用其可见 GPU；否则用 cuda:0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        ids = [s.strip() for s in visible.split(",") if s.strip() != ""]
    # 统一成 cuda:<id> 形式
    devices = [f"cuda:{i}" for i in ids]
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch URDF inference over cached test data (cache-dinov3-h-normalize style)."
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="cache",
        help="根目录，例如 /path/to/cache-dinov3-h-normalize",
    )
    parser.add_argument(
        "--model_path", type=str, default="/mnt/world_foundational_model/luoyulin/wzz_temp/urdf/V2/checkpoints/dinov3-h_900_urdf/lr1e-5_bs2_ep200_eot_urdf-params/epoch_200.pth",help="URDF 模型 checkpoint 路径"
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=os.path.join("urdf_anything", "model", "URDFModel_config.yaml"),
        help="URDF 模型配置文件路径",
    )
    parser.add_argument(
        "--dino_path",
        type=str,
        default="DINOv3",
        help="DINO 模型路径（与单张 inference 相同）",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1",
        help='逗号分隔的 GPU 编号，例如 "0,1"。默认读取 CUDA_VISIBLE_DEVICES 或用 cuda:0。',
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=512,
        help="TripoSG VAE token 数，与训练/inference.py 保持一致",
    )
    parser.add_argument(
        "--eot_threshold",
        type=float,
        default=0.5,
        help="EoT 判定阈值",
    )
    parser.add_argument(
        "--max_reconstruction_attempts",
        type=int,
        default=3,
        help="每个 link 的最大重建尝试次数",
    )
    parser.add_argument(
        "--overlap_chamfer_threshold",
        type=float,
        default=3e-2,
        help="相邻链接几何 Chamfer 距离小于该值视为高度重叠，停止生成",
    )

    args = parser.parse_args()

    cache_root = os.path.abspath(args.cache_root)
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"cache_root not found: {cache_root}")

    cache_paths = discover_cache_paths(cache_root)
    if not cache_paths:
        raise ValueError(f"No valid cache directories found under {cache_root}")

    print("Discovered cache directories:")
    for p in cache_paths:
        print(f"  - {p}")

    all_tasks: List[InferenceTask] = []
    for cp in cache_paths:
        tasks, meta = collect_tasks_from_cache(cp)
        print(
            f"Loaded test index from {cp}: "
            f"{len(tasks)} unique (obj_id, whole_image_name) pairs."
        )
        all_tasks.extend(tasks)

    if not all_tasks:
        print("No tasks collected from caches, exiting.")
        return

    devices = parse_devices_arg(args.devices)
    print(f"Using devices: {devices}")

    assignment = split_tasks_among_devices(all_tasks, devices)

    # 多进程并行，每块 GPU 一个子进程
    from multiprocessing import Process

    processes: List[Process] = []
    for dev in devices:
        dev_tasks = assignment.get(dev, [])
        p = Process(
            target=worker,
            args=(
                dev,
                dev_tasks,
                args.model_path,
                args.model_config_path,
                args.dino_path,
                args.num_tokens,
                args.eot_threshold,
                args.max_reconstruction_attempts,
                args.overlap_chamfer_threshold,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Batch inference completed.")


if __name__ == "__main__":
    main()

