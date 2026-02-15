"""Cache IO: load/save index, load single item, split train/test by object."""
import os
import json
import random
from datetime import datetime
import torch


def load_cache_data(cache_path, split="all"):
    """Load data index and metadata from cache directory."""
    metadata_path = os.path.join(cache_path, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if split == "train":
        data_path = os.path.join(cache_path, "train_data.pt")
    elif split == "test":
        data_path = os.path.join(cache_path, "test_data.pt")
    else:
        data_path = os.path.join(cache_path, "full_data.pt")
    data_index = torch.load(data_path, map_location="cpu")
    return data_index, metadata


def load_single_data_item(
    file_path,
    item_id=None,
    link_idx=None,
    whole_image_name=None,
):
    """Load one sample from a per-object cache .pt file."""
    obj_cache_data = torch.load(file_path, map_location="cpu")
    if item_id:
        for item in obj_cache_data["items"]:
            if item["id"] == item_id:
                if whole_image_name is None:
                    raise ValueError(
                        "需要提供 whole_image_name 来获取对应的 dino_features"
                    )
                result = item.copy()
                result["encode_whole"] = obj_cache_data["encode_whole"]
                result["dino_features"] = obj_cache_data["whole_image_features"][
                    whole_image_name
                ]
                result["whole_image_name"] = whole_image_name
                return result
        if obj_cache_data["eot_item"]["id"] == item_id:
            if whole_image_name is None:
                raise ValueError(
                    "whole_image_name is required to get corresponding dino_features"
                )
            result = obj_cache_data["eot_item"].copy()
            result["encode_whole"] = obj_cache_data["encode_whole"]
            result["dino_features"] = obj_cache_data["whole_image_features"][
                whole_image_name
            ]
            result["whole_image_name"] = whole_image_name
            return result
        raise ValueError(f"Item ID not found: {item_id}")

    if link_idx is not None and whole_image_name:
        if link_idx == obj_cache_data["eot_item"]["link_idx"]:
            result = obj_cache_data["eot_item"].copy()
            result["encode_whole"] = obj_cache_data["encode_whole"]
            result["dino_features"] = obj_cache_data["whole_image_features"][
                whole_image_name
            ]
            result["whole_image_name"] = whole_image_name
            return result
        for item in obj_cache_data["items"]:
            if item["link_idx"] == link_idx:
                result = item.copy()
                result["encode_whole"] = obj_cache_data["encode_whole"]
                result["dino_features"] = obj_cache_data["whole_image_features"][
                    whole_image_name
                ]
                result["whole_image_name"] = whole_image_name
                return result
        raise ValueError(
            f"Link index not found: link_idx={link_idx}, whole_image_name={whole_image_name}"
        )
    return obj_cache_data


def load_train_test_split(split_path):
    """从 JSON 加载 train / test 的 obj_id 列表。格式：{"train": [...], "test": [...]}。返回 (train_ids, test_ids) 均为 set(str)。"""
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_ids = set(str(x) for x in data.get("train", []))
    test_ids = set(str(x) for x in data.get("test", []))
    return train_ids, test_ids


def split_cache_by_object(cache_path, train_ratio=0.9, random_seed=42, train_test_split_path=None):
    """按物体划分 train/test 并写回。若提供 train_test_split_path，则直接使用 JSON 中的 train / test id 列表。"""
    print(f"\nre-split cache by object: {cache_path}")
    full_data_path = os.path.join(cache_path, "full_data.pt")
    if not os.path.exists(full_data_path):
        raise FileNotFoundError(f"data file not found: {full_data_path}")
    all_data = torch.load(full_data_path, map_location="cpu")
    if train_test_split_path and os.path.isfile(train_test_split_path):
        train_obj_ids, test_obj_ids = load_train_test_split(train_test_split_path)
        print(f"  using train/test split: {train_test_split_path} (train {len(train_obj_ids)}, test {len(test_obj_ids)} ids)")
        # 未出现在 JSON 中的 obj_id 划入 train
        for d in all_data:
            oid = str(d["obj_id"])
            if oid not in train_obj_ids and oid not in test_obj_ids:
                train_obj_ids.add(oid)
        train_data = [d for d in all_data if str(d["obj_id"]) in train_obj_ids]
        test_data = [d for d in all_data if str(d["obj_id"]) in test_obj_ids]
    else:
        random.seed(random_seed)
        unique_obj_ids = sorted(set(str(d["obj_id"]) for d in all_data))
        shuffled_obj_ids = unique_obj_ids.copy()
        random.shuffle(shuffled_obj_ids)
        total_obj_count = len(shuffled_obj_ids)
        train_obj_count = int(total_obj_count * train_ratio)
        train_obj_ids = set(shuffled_obj_ids[:train_obj_count])
        test_obj_ids = set(shuffled_obj_ids[train_obj_count:])
        train_data = [d for d in all_data if str(d["obj_id"]) in train_obj_ids]
        test_data = [d for d in all_data if str(d["obj_id"]) in test_obj_ids]
    torch.save(train_data, os.path.join(cache_path, "train_data.pt"))
    torch.save(test_data, os.path.join(cache_path, "test_data.pt"))
    metadata_path = os.path.join(cache_path, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    ratio = len(train_data) / len(all_data) if all_data else 0
    metadata.update(
        {
            "train_objects": len(train_obj_ids),
            "test_objects": len(test_obj_ids),
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "train_ratio": ratio if (train_test_split_path and os.path.isfile(train_test_split_path)) else train_ratio,
            "test_ratio": 1.0 - ratio if (train_test_split_path and os.path.isfile(train_test_split_path)) else (1.0 - train_ratio),
            "split_method": "train_test_split" if (train_test_split_path and os.path.isfile(train_test_split_path)) else "by_object",
            "random_seed": random_seed,
            "split_timestamp": datetime.now().isoformat(),
        }
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("re-split completed.")
    print(f"  - train samples: {len(train_data)}")
    print(f"  - test samples: {len(test_data)}")
    return train_data, test_data, metadata
