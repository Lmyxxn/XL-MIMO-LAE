# -*- coding: utf-8 -*-
import argparse, json, random
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import math

REQ_KEYS = ("H_list", "O_T_list", "IS_NEARFIELD_list")

# ----------------- Utilities -----------------

def to_object_array(py_list):
    arr = np.empty((len(py_list),), dtype=object)
    for i, v in enumerate(py_list):
        arr[i] = v
    return arr

def _safe_meta_load(data) -> dict:
    if "meta" in getattr(data, "files", []):
        try:
            v = data["meta"]
            if isinstance(v, np.ndarray) and v.dtype == object:
                v = v.item()
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            if isinstance(v, str):
                return json.loads(v)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    return {}

def _parse_center_arg(s: Union[str, None]) -> Union[np.ndarray, None]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--upa-center requires format 'x,y,z'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)

def _extract_center_from_meta(meta: Dict[str, Any]) -> Union[np.ndarray, None]:
    keys = ["upa_center", "UPA_center", "center", "array_center"]
    for k in keys:
        if k in meta:
            v = meta[k]
            try:
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
                if isinstance(v, dict) and all(ax in v for ax in ("x", "y", "z")):
                    return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float32)
            except Exception:
                pass
    return None

def _angles_and_distance(POS: np.ndarray, center: np.ndarray) -> np.ndarray:
    POS = np.asarray(POS, dtype=np.float32)
    if POS.ndim != 2 or POS.shape[-1] != 3:
        raise ValueError(f"POS expected [T,3], got {POS.shape}")
    R = POS - center.reshape(1, 3)
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    dist = np.linalg.norm(R, axis=1)
    eps = 1e-9
    safe = np.maximum(dist, eps)
    theta = np.arccos(np.clip(z / safe, -1.0, 1.0)).astype(np.float32)
    phi   = np.arctan2(y, x).astype(np.float32)
    return np.stack([theta, phi, dist.astype(np.float32)], axis=-1)

def _compute_angle_for_structure(pos_arr, center: np.ndarray):
    # O_T_list may be object or numeric array; unify return as "one element per entry"
    if isinstance(pos_arr, np.ndarray) and pos_arr.dtype == object:
        out = []
        for it in pos_arr:
            v = it
            if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
                v = v.item()
            out.append(_angles_and_distance(v, center))
        return to_object_array(out)

    pos_arr = np.asarray(pos_arr)
    if pos_arr.ndim == 2 and pos_arr.shape[-1] == 3:
        return _angles_and_distance(pos_arr, center)
    elif pos_arr.ndim >= 3 and pos_arr.shape[-1] == 3:
        out = [_angles_and_distance(pos_arr[i], center) for i in range(pos_arr.shape[0])]
        return to_object_array(out)
    else:
        raise ValueError(f"O_T_list shape not supported: {pos_arr.shape}")

# ----------------- Main Logic -----------------

def scan_files(root: Path, pattern="*_channels.npz") -> List[Path]:
    files = sorted(root.rglob(pattern))
    excl = {"merged_channels.npz", "train.npz", "val.npz", "test.npz",
            "train_all.npz", "val_all.npz", "test_all.npz"}
    files = [p for p in files if p.name not in excl]
    return files

def load_file_as_entry(p: Path, center_arg: Union[np.ndarray, None]) -> Dict[str, Any]:
    data = np.load(p, allow_pickle=True)
    if not all(k in data.files for k in REQ_KEYS):
        raise KeyError(f"{p.name} missing required keys {REQ_KEYS}")

    fname = p.stem
    if "__" in fname:
        main_scene = fname.split("__", 1)[0]   # e.g. scene_1
    else:
        main_scene = fname[:-len("_channels")] if fname.endswith("_channels") else fname

    meta = _safe_meta_load(data)
    center = center_arg if center_arg is not None else (_extract_center_from_meta(meta) or np.zeros(3, dtype=np.float32))

    O_T_list = data["O_T_list"]
    Angle_list = _compute_angle_for_structure(O_T_list, center)

    return dict(
        file=str(p),
        main_scene=main_scene,
        H_list=data["H_list"],
        O_T_list=O_T_list,
        IS_NEARFIELD_list=data["IS_NEARFIELD_list"],
        Angle_Distance_list=Angle_list,
        meta=meta
    )

def split_by_scene_ratio(
    entries: List[Dict[str, Any]],
    seed: int,
    train_ratio: float | None,
    val_ratio: float,
    test_ratio: float,
    min_keep_per_scene: int = 1,
):
    """
    Split each main_scene independently (mutually exclusive):
      - If train_ratio is None, then train = max(0, 1 - val - test).
      - Each ratio is calculated as ceil(n*ratio); then clipped to not exceed n, and ensure their sum does not exceed n.
      - Take val first, then test, and finally train.
    """
    rng = random.Random(seed)
    groups: Dict[str, List[int]] = {}
    for i, e in enumerate(entries):
        groups.setdefault(e["main_scene"], []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for scene, idxs in groups.items():
        idxs = idxs[:]  # copy
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 0:
            continue

        tr = (1.0 - val_ratio - test_ratio) if train_ratio is None else float(train_ratio)
        tr = max(0.0, tr)
        vr = max(0.0, float(val_ratio))
        ter = max(0.0, float(test_ratio))

        # Ceil each ratio
        k_val  = int(math.ceil(vr  * n))
        k_test = int(math.ceil(ter * n))
        k_train= int(math.ceil(tr  * n))

        # Clip to not exceed n, and ensure sum does not exceed n
        k_val  = min(k_val, n)
        k_test = min(k_test, n - k_val)
        k_train= min(k_train, n - k_val - k_test)

        # All zero and n>0, keep min_keep_per_scene for train
        if (k_val + k_test + k_train) == 0 and n > 0:
            k_train = min_keep_per_scene

        # Split: val -> test -> train
        v_sel  = idxs[:k_val]
        t_sel  = idxs[k_val:k_val+k_test]
        tr_sel = idxs[k_val+k_test:k_val+k_test+k_train]

        val_idx  += v_sel
        test_idx += t_sel
        train_idx+= tr_sel

    return train_idx, val_idx, test_idx

def save_entries(entries: List[Dict[str, Any]], out_npz: Path, tag: str):
    H_obj    = to_object_array([e["H_list"] for e in entries])
    POS_obj  = to_object_array([e["O_T_list"] for e in entries])
    ANG_obj  = to_object_array([e["Angle_Distance_list"] for e in entries])
    NF_obj   = to_object_array([e["IS_NEARFIELD_list"] for e in entries])
    meta_obj = to_object_array([e.get("meta", {}) for e in entries])
    id_obj   = to_object_array([(e["main_scene"], e["file"]) for e in entries])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        H_list=H_obj,
        O_T_list=POS_obj,
        Angle_Distance=ANG_obj,
        IS_NEARFIELD_list=NF_obj,
        meta_list=meta_obj,
        ids=id_obj,
        info=json.dumps({"tag": tag, "num_files": len(entries)}, ensure_ascii=False)
    )
    print(f"[OK] {tag}: {len(entries)} files -> {out_npz}")

# ----------------- Batch process one main scene (split and save in its own data/) -----------------

def process_one_scene(scene_dir: Path, train_ratio: float | None, val_ratio: float, test_ratio: float,
                      seed: int, center_arg: Union[np.ndarray, None]):
    data_dir = scene_dir / "data"
    if not data_dir.exists():
        print(f"[WARN] Skip: {data_dir} does not exist")
        return

    print(f"\n=== Processing main scene: {scene_dir.name} ===")
    files = scan_files(data_dir)
    if not files:
        print(f"[WARN] No *_channels.npz found in {data_dir}, skipping.")
        return

    # Load
    entries = []
    for f in files:
        try:
            entries.append(load_file_as_entry(f, center_arg))
        except Exception as e:
            print(f"[WARN] Skip {f}: {e}")
    if not entries:
        print("[WARN] No valid entries, skipping.")
        return

    # Split
    train_idx, val_idx, test_idx = split_by_scene_ratio(
        entries,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_keep_per_scene=0
    )
    train_set = [entries[i] for i in train_idx]
    val_set   = [entries[i] for i in val_idx]
    test_set  = [entries[i] for i in test_idx]

    # Save to the main scene's own data/ directory
    if (train_ratio is None and (1 - val_ratio - test_ratio) > 0) or (train_ratio and train_ratio > 0):
        if len(train_set) > 0:
            save_entries(train_set, data_dir / "train.npz", "TRAIN")
    if val_ratio > 0 and len(val_set) > 0:
        save_entries(val_set,   data_dir / "val.npz",   "VAL")
    if test_ratio > 0 and len(test_set) > 0:
        save_entries(test_set,  data_dir / "test.npz",  "TEST")

    print(f"[DONE:{scene_dir.name}] sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(
        description="Split the data/ directory of each main scene scene_* under the parent directory by ratio (sets with ratio 0 are not saved)."
    )
    ap.add_argument("--parent-root", default='./DownstreamDataset/Hybrid_Channel_UPA64X64/ValSet/Urban/Nanjing',
                    help="Parent directory containing multiple main scenes scene_*")
    ap.add_argument("--train-ratio", type=float, default=0.0,
                    help="Training set ratio (0~1); if None, use 1 - val - test (cannot pass None via command line, can leave 0 and calculate manually)")
    ap.add_argument("--val-ratio",   type=float, default=1.0,  help="Validation set ratio (0~1)")
    ap.add_argument("--test-ratio",  type=float, default=0.0,  help="Test set ratio (0~1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--upa-center", type=str, default='0,0,35',
                    help="UPA center 'x,y,z'; if not provided, read from meta first, otherwise use 0,0,0")
    args = ap.parse_args()

    parent = Path(args.parent_root)
    if not parent.exists():
        raise SystemExit(f"[ERR] Parent directory does not exist: {parent}")

    # Basic checks
    for name, r in [("train-ratio", args.train_ratio if args.train_ratio is not None else 1.0 - args.val_ratio - args.test_ratio),
                    ("val-ratio", args.val_ratio),
                    ("test-ratio", args.test_ratio)]:
        if r < 0 or r > 1:
            raise SystemExit(f"[ERR] {name} must be in [0,1], got {r}")

    if args.train_ratio is None and (args.val_ratio + args.test_ratio) > 1.0 + 1e-8:
        raise SystemExit("[ERR] val-ratio + test-ratio > 1, cannot automatically assign train ratio")

    center_arg = _parse_center_arg(args.upa_center)

    # Iterate over each main scene scene_*, perform splitting and saving in its own data/ directory
    scene_dirs = sorted([d for d in parent.glob("scene_*") if d.is_dir()])
    if not scene_dirs:
        raise SystemExit(f"[ERR] No scene_* directories found under {parent}")

    for scene_dir in scene_dirs:
        process_one_scene(scene_dir,
                          train_ratio=args.train_ratio,
                          val_ratio=args.val_ratio,
                          test_ratio=args.test_ratio,
                          seed=args.seed,
                          center_arg=center_arg)

if __name__ == "__main__":
    main()
