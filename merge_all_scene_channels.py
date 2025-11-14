#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional

# ---------- Utilities ----------

def object_array(py_list):
    arr = np.empty((len(py_list),), dtype=object)
    for i, it in enumerate(py_list):
        arr[i] = it
    return arr

def try_stack(arr_list: List[np.ndarray]) -> Tuple[np.ndarray, bool]:
    try:
        return np.stack(arr_list, axis=0), True
    except Exception:
        return object_array(arr_list), False

def get_first_existing_key(d: Any, keys: List[str]) -> Optional[np.ndarray]:
    names = set(getattr(d, "files", []))
    for k in keys:
        if k in names:
            return d[k]
    return None

# ---------- Shape Predicates ----------

def _pred_H(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.ndim >= 3 and x.shape[-1] == 2

def _pred_POS(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.ndim >= 2 and x.shape[-1] == 3

def _pred_ANG(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.ndim >= 2 and x.shape[-1] == 3

def _pred_ISN(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.ndim >= 1 and x.dtype != object

def _flatten_to_items(arr: Any, pred: Callable[[Any], bool]) -> List[np.ndarray]:
    """
    Recursively flatten a possibly multi-layered object array, extract all ndarrays satisfying pred,
    and return a flat list of np.ndarray.
    """
    out: List[np.ndarray] = []

    def rec(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            for it in x:
                y = it
                if isinstance(y, np.ndarray) and y.dtype == object and y.size == 1:
                    y = y.item()
                rec(y)
        elif isinstance(x, np.ndarray) and not pred(x) and x.ndim >= 1 and x.dtype != object:
            for i in range(x.shape[0]):
                rec(x[i])
        else:
            if pred(x):
                out.append(x)

    rec(arr)
    return out

def _is_valid_frame(H_t_ri: np.ndarray) -> bool:
    # H_t_ri: [M,K,2]
    return isinstance(H_t_ri, np.ndarray) and H_t_ri.ndim == 3 and H_t_ri.shape[-1] == 2 and np.isfinite(H_t_ri).all()

# ---------- Aggregation and "Frame-wise Densification" of a Split ----------

def gather_split_dense(
    root_dir: str | Path,
    split: str,
    require_nf: bool = False,
    skip_mismatch: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Scan root/scene_*/data/{split}.npz, recursively flatten to individual items, then expand along the T dimension to "frame-wise",
    concatenate into dense arrays and count near-field occurrences. ANGLE is treated the same as H/POS/ISNF: always output [N,3], missing values are filled with NaN.
    """
    root = Path(root_dir)
    scenes = sorted([p for p in root.glob("scene_*") if p.is_dir()])
    if not scenes:
        raise RuntimeError(f"No scene_* directories found under {root}")

    frames_H:   List[np.ndarray] = []
    frames_POS: List[np.ndarray] = []
    frames_ISN: List[int]        = []
    frames_ANG: List[np.ndarray] = []
    index_tbl:  List[tuple]      = []

    # Pre-detect global M,K
    M_glob = K_glob = None
    for scene_dir in scenes:
        npz_path = scene_dir / "data" / f"{split}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        H_arr = get_first_existing_key(data, ["H_list", "H_ALL", "H_ALL_scene"])
        if H_arr is None:
            continue
        H_items = _flatten_to_items(H_arr, _pred_H)
        if not H_items:
            continue
        H0 = H_items[0]  # Expected shape [T,M,K,2]
        if isinstance(H0, np.ndarray) and H0.ndim == 4 and H0.shape[-1] == 2:
            M_glob, K_glob = H0.shape[1], H0.shape[2]
            break
    if M_glob is None or K_glob is None:
        raise RuntimeError("Unable to infer global (M,K) from input. Please check the data.")

    missing_ang_frames = 0

    # Gather
    for si, scene_dir in enumerate(scenes):
        npz_path = scene_dir / "data" / f"{split}.npz"
        if not npz_path.exists():
            print(f"[WARN] Skipping (no such split): {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        H_arr   = get_first_existing_key(data, ["H_list", "H_ALL", "H_ALL_scene"])
        POS_arr = get_first_existing_key(data, ["O_T_list", "POS_ALL", "POS_ALL_scene"])
        ISN_arr = get_first_existing_key(data, ["IS_NEARFIELD_list", "IS_NF_ALL", "ISNF_ALL_scene"])
        ANG_arr = get_first_existing_key(data, ["Angle_Distance", "Angle_Distance_list", "ANGLE_ALL", "ANGLE_ALL_scene"])

        if H_arr is None or POS_arr is None or ISN_arr is None:
            print(f"[WARN] Missing required keys, skipping: {npz_path}")
            continue

        H_items   = _flatten_to_items(H_arr,   _pred_H)
        POS_items = _flatten_to_items(POS_arr, _pred_POS)
        ISN_items = _flatten_to_items(ISN_arr, _pred_ISN)
        ANG_items = _flatten_to_items(ANG_arr, _pred_ANG) if ANG_arr is not None else []

        n_items = min(len(H_items), len(POS_items), len(ISN_items))
        H_items, POS_items, ISN_items = H_items[:n_items], POS_items[:n_items], ISN_items[:n_items]

        for ii in range(n_items):
            H_it, P_it, N_it = H_items[ii], POS_items[ii], ISN_items[ii]
            A_it = ANG_items[ii] if ii < len(ANG_items) else None

            # Quick shape validation
            if not (isinstance(H_it, np.ndarray) and H_it.ndim == 4 and H_it.shape[-1] == 2):
                print(f"[WARN] Invalid H shape, skipping scene={si}, item={ii}: {getattr(H_it, 'shape', None)}")
                continue
            if not (isinstance(P_it, np.ndarray) and P_it.ndim >= 2 and P_it.shape[-1] == 3):
                print(f"[WARN] Invalid POS shape, skipping scene={si}, item={ii}: {getattr(P_it, 'shape', None)}")
                continue
            if not (isinstance(N_it, np.ndarray) and N_it.ndim >= 1 and N_it.dtype != object):
                print(f"[WARN] Invalid ISNF shape, skipping scene={si}, item={ii}: {getattr(N_it, 'shape', None)}")
                continue
            if A_it is not None and not (isinstance(A_it, np.ndarray) and A_it.ndim >= 2 and A_it.shape[-1] == 3):
                # Treat the entire ANG item as missing
                A_it = None

            T = H_it.shape[0]
            for ti in range(T):
                H_t = H_it[ti]  # [M,K,2]
                if not _is_valid_frame(H_t):
                    continue

                if H_t.shape[0] != M_glob or H_t.shape[1] != K_glob:
                    if skip_mismatch:
                        continue
                    raise RuntimeError(f"M/K mismatch: {(H_t.shape[0], H_t.shape[1])} vs {(M_glob, K_glob)} @ (scene,item,t)=({si},{ii},{ti})")

                isnf = int(N_it[ti]) if ti < len(N_it) else 1
                if require_nf and isnf != 1:
                    continue

                pos_t = P_it[ti]
                if pos_t.shape != (3,):
                    continue

                # angle: use NaN(3,) as placeholder if missing or invalid
                if A_it is None or ti >= len(A_it):
                    ang_t = np.full((3,), np.nan, dtype=np.float32)
                    missing_ang_frames += 1
                else:
                    a = A_it[ti].reshape(-1)
                    if not (isinstance(a, np.ndarray) and a.shape[-1] == 3 and np.isfinite(a).all()):
                        ang_t = np.full((3,), np.nan, dtype=np.float32)
                        missing_ang_frames += 1
                    else:
                        ang_t = a.astype(np.float32, copy=False)

                frames_H.append(H_t.astype(np.float32, copy=False))
                frames_POS.append(pos_t.astype(np.float32, copy=False))
                frames_ISN.append(isnf)
                frames_ANG.append(ang_t)
                index_tbl.append((si, ii, ti))

    if not frames_H:
        raise RuntimeError(f"{split}: No valid frames collected (filters too strict or data issue).")

    # Build dense arrays
    H_dense     = np.stack(frames_H,   axis=0).astype(np.float32)   # [N,M,K,2]
    POS_dense   = np.stack(frames_POS, axis=0).astype(np.float32)   # [N,3]
    ISNF_dense  = np.asarray(frames_ISN, dtype=np.int8)             # [N]
    ANG_dense   = np.stack(frames_ANG, axis=0).astype(np.float32)   # [N,3]
    INDEX_dense = np.asarray(index_tbl, dtype=np.int32)             # [N,3]

    # ---- Near-field statistics ----
    nf_count  = int(np.sum(ISNF_dense == 1))
    far_count = int(np.sum(ISNF_dense == 0))
    total     = int(ISNF_dense.size)
    nf_ratio  = (nf_count / total) if total > 0 else 0.0
    miss_ratio = float(missing_ang_frames) / float(ANG_dense.shape[0]) if ANG_dense.size else 0.0

    print(f"[NF]    split={split}: near={nf_count}, far={far_count}, total={total}, ratio={nf_ratio:.4f}")
    print(f"[ANGLE] split={split}: missing_frames={missing_ang_frames}/{ANG_dense.shape[0]} ({miss_ratio:.2%})")

    out = dict(
        H_ALL_scene_dense=H_dense,
        POS_ALL_scene_dense=POS_dense,
        ISNF_ALL_scene_dense=ISNF_dense,
        ANGLE_ALL_scene_dense=ANG_dense,
        INDEX_scene_dense=INDEX_dense,
        # Statistics (stored in package, scalars as 1-element arrays)
        NF_COUNT=np.asarray([nf_count], dtype=np.int32),
        FAR_COUNT=np.asarray([far_count], dtype=np.int32),
        TOTAL_COUNT=np.asarray([total], dtype=np.int32),
        M=np.asarray([H_dense.shape[1]], dtype=np.int32),
        K=np.asarray([H_dense.shape[2]], dtype=np.int32),
    )
    return out

# ---------- Entry point ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Aggregate train/val/test of scene_* and flatten nested structure into frame-wise dense storage (N,M,K,2)."
    )
    ap.add_argument("--root",   default="./DownstreamDataset/Hybrid_Channel_UPA64X64/ValSet/Urban/Nanjing", help="Root directory, e.g., Dataset_LAE_64X64")
    ap.add_argument("--outdir", default="./DownstreamDataset/Hybrid_Channel_UPA64X64/ValSet/Urban/Nanjing", help="Output directory (default is --root)")
    ap.add_argument("--require-nf", action="store_true", default=False, help="Keep only near-field frames (ISNF==1)")
    ap.add_argument("--no-skip-mismatch", dest="skip_mismatch", action="store_false",
                    help="Raise error on M/K mismatch (default skips mismatched frames)")

    # Select splits to aggregate (default: train enabled, val/test disabled)
    ap.add_argument("--do-train", dest="do_train", action="store_true", default=False,  help="Aggregate train split")
    ap.add_argument("--do-val",   dest="do_val",   action="store_true", default=True, help="Aggregate val split")
    ap.add_argument("--do-test",  dest="do_test",  action="store_true", default=False, help="Aggregate test split")

    # Only report near-field count, do not save output
    ap.add_argument("--report-nf-only", action="store_true", default=False,
                    help="Only report and print near-field count, do not save output files")

    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(args.root)
    outdir.mkdir(parents=True, exist_ok=True)

    print(args.root)

    if not args.do_train and not args.do_val and not args.do_test:
        print("[INFO] No splits selected for aggregation (train/val/test); exiting.")
        raise SystemExit(0)

    def maybe_save(bundle: Dict[str, np.ndarray], path: Path, split_name: str):
        if not args.report_nf_only:
            np.savez_compressed(path, **bundle)
            print(f"[SAVED] {split_name}_all_dense -> {path}")
        else:
            near = int(bundle["NF_COUNT"][0]); far = int(bundle["FAR_COUNT"][0]); total = int(bundle["TOTAL_COUNT"][0])
            ratio = near / total if total > 0 else 0.0
            print(f"[REPORT] {split_name}: near={near}, far={far}, total={total}, ratio={ratio:.4f}")

    if args.do_train:
        train_bundle = gather_split_dense(args.root, "train",
                                          require_nf=args.require_nf,
                                          skip_mismatch=args.skip_mismatch)
        maybe_save(train_bundle, outdir / "train_all_dense.npz", "train")

    if args.do_val:
        val_bundle = gather_split_dense(args.root, "val",
                                        require_nf=args.require_nf,
                                        skip_mismatch=args.skip_mismatch)
        maybe_save(val_bundle, outdir / "val_all_dense.npz", "val")

    if args.do_test:
        test_bundle = gather_split_dense(args.root, "test",
                                         require_nf=args.require_nf,
                                         skip_mismatch=args.skip_mismatch)
        maybe_save(test_bundle, outdir / "test_all_dense.npz", "test")
