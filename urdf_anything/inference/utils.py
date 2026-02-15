"""Inference utilities: image resize, mesh watertight."""
import torch
import torch.nn.functional as F
import trimesh


def resize_image_tensor(img, target_size=(512, 512)):
    """Resize image tensor to target_size (H, W)."""
    if img.dim() == 3:
        if img.shape[2] == 3:
            img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img_resized = F.interpolate(
        img, size=target_size, mode="bilinear", align_corners=False
    )
    img_resized = img_resized.squeeze(0).permute(1, 2, 0)
    return img_resized


def ensure_watertight(mesh):
    """Try to make mesh watertight; return as-is if not trimesh.Trimesh."""
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
            try:
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
            except AttributeError:
                mesh.process(validate=True)
        except Exception as e:
            print(f"warning: failed to repair watertightness: {e}")
    return mesh
