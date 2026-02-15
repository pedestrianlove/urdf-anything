"""TripoSG 3D generation from image: load script by path, run inference."""
import os
import sys
import importlib.util
import numpy as np
import torch
import trimesh

# Resolve repo root (codes/) and TripoSG dir
_CODES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRIPOSG_DIR = os.path.join(_CODES_DIR, "TripoSG")

_TRIPOSG_PIPELINE = None
_TRIPOSG_RMBG = None


def get_3d_triposg(
    image_path,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    target_faces: int = -1,
    triposg_weights_dir: str | None = None,
    rmbg_weights_dir: str | None = None,
    fix_normals: bool = True,
):
    """Generate 3D mesh from image using TripoSG (image-to-3D)."""
    global _TRIPOSG_PIPELINE, _TRIPOSG_RMBG
    try:
        if _TRIPOSG_DIR not in sys.path:
            sys.path.insert(0, _TRIPOSG_DIR)
        _script_path = os.path.join(_TRIPOSG_DIR, "scripts", "inference_triposg.py")
        if not os.path.isfile(_script_path):
            raise ImportError(f"TripoSG script not found: {_script_path}")
        _spec = importlib.util.spec_from_file_location(
            "inference_triposg_triposg", _script_path
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        run_triposg = _mod.run_triposg
        TripoSGPipeline = _mod.TripoSGPipeline
        BriaRMBG = _mod.BriaRMBG
    except ImportError as e:
        err = str(e)
        if "libGL" in err or "libgl" in err.lower():
            raise ImportError(
                f"TripoSG 依赖 cv2 (OpenCV)，当前环境缺少 libGL: {err}. "
                "解决：安装系统库 `libgl1-mesa-glx` 或 `pip install opencv-python-headless`。"
            ) from e
        raise ImportError(
            f"Failed to import TripoSG module: {e}. "
            "Please ensure TripoSG is in the project root (codes/TripoSG)."
        ) from e

    if triposg_weights_dir is None:
        triposg_weights_dir = os.path.join(
            _CODES_DIR, "TripoSG", "pretrained_weights", "TripoSG"
        )
    if rmbg_weights_dir is None:
        rmbg_weights_dir = os.path.join(
            _CODES_DIR, "TripoSG", "pretrained_weights", "RMBG-1.4"
        )

    if _TRIPOSG_RMBG is None:
        print(f"Loading RMBG model: {rmbg_weights_dir}")
        _TRIPOSG_RMBG = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        _TRIPOSG_RMBG.eval()
    if _TRIPOSG_PIPELINE is None:
        print(f"Loading TripoSG pipeline: {triposg_weights_dir}")
        _TRIPOSG_PIPELINE = TripoSGPipeline.from_pretrained(
            triposg_weights_dir
        ).to(device, dtype)

    rmbg_net = _TRIPOSG_RMBG
    pipe = _TRIPOSG_PIPELINE
    print("Running TripoSG inference (run_triposg)...")
    mesh = run_triposg(
        pipe=pipe,
        image_input=image_path,
        rmbg_net=rmbg_net,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        faces=target_faces,
    )

    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            vertices = (
                mesh.vertices.cpu().numpy()
                if isinstance(mesh.vertices, torch.Tensor)
                else mesh.vertices
            )
            faces = (
                mesh.faces.cpu().numpy()
                if isinstance(mesh.faces, torch.Tensor)
                else mesh.faces
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise ValueError(
                f"Cannot convert mesh to trimesh.Trimesh: {type(mesh)}"
            )
    mesh.export(os.path.join(_CODES_DIR, "output", "mesh_for_edit.obj"))

    print("\n可选：在终端中对 mesh 做简单旋转。可多次旋转，直接回车结束。")
    print("  0: 不旋转（直接回车也等价于结束）")
    print("  1-6: 围绕 X/Y/Z 轴 ±90°")
    angle_map = {
        "1": (np.pi / 2, [1, 0, 0]),
        "2": (-np.pi / 2, [1, 0, 0]),
        "3": (np.pi / 2, [0, 1, 0]),
        "4": (-np.pi / 2, [0, 1, 0]),
        "5": (np.pi / 2, [0, 0, 1]),
        "6": (-np.pi / 2, [0, 0, 1]),
    }
    while True:
        try:
            choice = input(
                "输入旋转编号并回车（直接回车=确认并结束旋转）[0-6]: "
            ).strip()
        except EOFError:
            choice = "0"
        if choice == "" or choice == "0":
            break
        if choice in angle_map:
            angle, axis_vec = angle_map[choice]
            R = trimesh.transformations.rotation_matrix(angle, axis_vec)
            mesh.apply_transform(R)
            mesh.export(
                os.path.join(_CODES_DIR, "output", "mesh_for_edit.obj")
            )
        else:
            print("输入无效，请输入 0–6，或直接回车结束。")

    bmin, bmax = mesh.bounds
    size = bmax - bmin
    if np.all(size > 0):
        center = (bmin + bmax) / 2.0
        scale = float(2.0 / np.max(size))
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
    print("TripoSG generation completed")
    return mesh
