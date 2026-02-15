"""Model utilities: load OBJ surface, load YAML/JSON config."""
import os
import json
import yaml
import numpy as np
import torch
import trimesh


def load_obj_surface(obj_path, num_pc=204800):
    """Load mesh from OBJ, sample surface points with normals, return tensor [1, N, 6]."""
    mesh = trimesh.load(obj_path)
    if not mesh.is_watertight:
        mesh.fill_holes()
        try:
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
        except AttributeError:
            mesh.process(validate=True)
    surface_points, face_indices = mesh.sample(num_pc, return_index=True)
    faces = mesh.faces[face_indices]
    face_normals = mesh.face_normals[face_indices]
    point_normals = face_normals
    surface_points = np.array(surface_points, dtype=np.float32)
    point_normals = np.array(point_normals, dtype=np.float32)
    surface_tensor = torch.FloatTensor(surface_points)
    normal_tensor = torch.FloatTensor(point_normals)
    surface_with_normals = torch.cat([surface_tensor, normal_tensor], dim=-1)
    surface_with_normals = surface_with_normals.unsqueeze(0)
    return surface_with_normals


def load_model_config(config_path):
    """Load model config from YAML or JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {config_path}, please use .yaml or .json"
        )
    return config


def get_default_config_path():
    """Return absolute path to default URDFModel config (in this package)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "URDFModel_config.yaml")
    )
