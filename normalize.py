import os
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def compute_normalization_from_whole(whole_obj_path: str):
    """Compute uniform scaling and translation from whole.obj to fit into [-1, 1]^3"""
    mesh = trimesh.load(whole_obj_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{whole_obj_path} is not a valid mesh")

    bmin, bmax = mesh.bounds
    size = bmax - bmin
    if not np.all(size > 0):
        raise ValueError(f"Bounding box size of {whole_obj_path} is 0, cannot normalize")

    center = (bmin + bmax) / 2.0
    # Uniform scaling: scale the longest edge to length 2, ensuring the whole object fits into [-1, 1]^3
    scale = float(2.0 / np.max(size))
    return center, scale


def transform_vertices(mesh: trimesh.Trimesh, center: np.ndarray, scale: float):
    """Apply translation + uniform scaling to mesh (centered at origin)"""
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    return mesh


def transform_vec_string(vec_str: str, center: np.ndarray, scale: float) -> str:
    """Apply the same translation + scaling to a string of the form 'x y z'"""
    vals = np.fromstring(vec_str, sep=" ", dtype=np.float64)
    if vals.shape[0] != 3:
        return vec_str
    vals = (vals - center) * scale
    return " ".join(f"{v:.8f}" for v in vals)


def normalize_info_json(src_path: str, dst_path: str, center: np.ndarray, scale: float):
    with open(src_path, "r") as f:
        info = json.load(f)

    # origin_xyz under links needs the same coordinate transformation
    for link in info.get("links", []):
        if "origin_xyz" in link:
            link["origin_xyz"] = transform_vec_string(link["origin_xyz"], center, scale)
        # axis_xyz is a direction vector, uniform scaling does not change direction, keep it unchanged

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(info, f, indent=2)


def normalize_test_urdf(src_path: str, dst_path: str, center: np.ndarray, scale: float):
    tree = ET.parse(src_path)
    root = tree.getroot()

    # Step 1: Apply linear coordinate transformation to all origin elements (unified coordinate system)
    for origin in root.iter("origin"):
        xyz = origin.get("xyz")
        if xyz is not None:
            origin.set("xyz", transform_vec_string(xyz, center, scale))

    # Step 2: Based on joint's origin, force child link's origin = -joint origin
    # This ensures that "other joints' link.origin always equals the negative of the corresponding joint.origin"

    # First, find all link names and which links appear as children of joints, to identify root links
    link_names = {link.get("name") for link in root.iter("link") if link.get("name") is not None}
    joint_infos = []
    child_links = set()

    for joint in root.iter("joint"):
        origin_el = joint.find("origin")
        if origin_el is None:
            continue
        xyz = origin_el.get("xyz")
        if xyz is None:
            continue
        vals = np.fromstring(xyz, sep=" ", dtype=np.float64)
        if vals.shape[0] != 3:
            continue

        parent_el = joint.find("parent")
        child_el = joint.find("child")
        parent_name = parent_el.get("link") if parent_el is not None else None
        child_name = child_el.get("link") if child_el is not None else None
        if child_name is not None:
            child_links.add(child_name)

        joint_infos.append(
            {
                "joint": joint,
                "origin_el": origin_el,
                "origin_vec": vals,
                "parent": parent_name,
                "child": child_name,
            }
        )

    root_links = link_names - child_links  # Links that are parents are considered base links

    # For all joints:
    # - If parent is a root link (base -> first link joint), force:
    #     joint.origin = 0 0 0, and its child link's visual/collision.origin also = 0 0 0
    # - Otherwise, force child link.origin = -joint.origin
    for info in joint_infos:
        parent = info["parent"]
        child = info["child"]
        joint_origin = info["origin_vec"]

        # base -> first link joint:
        # parent is a root link, set joint.origin to 0 by convention,
        # and the origin inside the first link it connects is also set to 0
        if parent in root_links:
            info["origin_el"].set("xyz", "0 0 0")
            if child is not None:
                for link in root.iter("link"):
                    if link.get("name") != child:
                        continue
                    for tag in ("visual", "collision"):
                        for elem in link.findall(tag):
                            origin_el = elem.find("origin")
                            if origin_el is not None:
                                origin_el.set("xyz", "0 0 0")
            continue

        if child is None:
            continue

        # Non-root joint: find child link and modify its visual/collision origin = -joint.origin
        for link in root.iter("link"):
            if link.get("name") != child:
                continue

            neg = -joint_origin
            neg_str = " ".join(f"{v:.8f}" for v in neg)

            for tag in ("visual", "collision"):
                for elem in link.findall(tag):
                    origin_el = elem.find("origin")
                    if origin_el is not None:
                        origin_el.set("xyz", neg_str)


    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tree.write(dst_path, encoding="utf-8", xml_declaration=True)


def normalize_obj(src_path: str, dst_path: str, center: np.ndarray, scale: float):
    mesh = trimesh.load(src_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{src_path} is not a valid mesh")
    mesh = transform_vertices(mesh, center, scale)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    mesh.export(dst_path)


def process_object_dir(src_obj_dir: str, dst_obj_dir: str):
    """Process a single object id directory"""
    whole_obj = os.path.join(src_obj_dir, "whole.obj")
    if not os.path.exists(whole_obj):
        # Some directories may not be standard object directories, copy structure directly without normalization
        shutil.copytree(src_obj_dir, dst_obj_dir, dirs_exist_ok=True)
        return

    center, scale = compute_normalization_from_whole(whole_obj)

    os.makedirs(dst_obj_dir, exist_ok=True)

    for name in os.listdir(src_obj_dir):
        src_path = os.path.join(src_obj_dir, name)
        dst_path = os.path.join(dst_obj_dir, name)

        if os.path.isdir(src_path):
            # If there are subdirectories (e.g., images), copy recursively
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            continue

        if name == "mobility.urdf":
            continue

        if name.endswith(".py") or name.endswith(".ipynb"):
            continue

        if name.endswith(".obj"):
            normalize_obj(src_path, dst_path, center, scale)
            continue

        if name == "info.json":
            normalize_info_json(src_path, dst_path, center, scale)
            continue

        if name == "test.urdf":
            normalize_test_urdf(src_path, dst_path, center, scale)
            continue

        # Other files (images, gifs, etc.) are copied as-is
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)


def main():
    codes_dir = os.path.dirname(os.path.abspath(__file__))
    processed_root = os.path.dirname(codes_dir)
    project_root = os.path.dirname(processed_root)
    normalized_root = os.path.join(project_root, "data_normalized")

    os.makedirs(normalized_root, exist_ok=True)

    for category in os.listdir(processed_root):
        if category == "Faucet":
            continue
        src_cat_dir = os.path.join(processed_root, category)
        if not os.path.isdir(src_cat_dir):
            continue
        if category == "codes":
            continue

        dst_cat_dir = os.path.join(normalized_root, category)
        os.makedirs(dst_cat_dir, exist_ok=True)

        # Subdirectories under each category directory are generally obj_id
        for obj_id in os.listdir(src_cat_dir):
            src_obj_dir = os.path.join(src_cat_dir, obj_id)
            if not os.path.isdir(src_obj_dir):
                continue
            dst_obj_dir = os.path.join(dst_cat_dir, obj_id)
            print(f"Normalizing {category}/{obj_id} -> {os.path.relpath(dst_obj_dir, project_root)}")
            process_object_dir(src_obj_dir, dst_obj_dir)


if __name__ == "__main__":
    main()