# -*- coding: utf-8 -*-
# —— Basic configuration ——
colab_compat = False
resolution = [480, 320]

class ExitCell(Exception):
    def _render_traceback_(self):
        pass

import os
gpu_num = 0  # Use CPU if set to ""; use GPU 0 if set to 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# —— TensorFlow initialization —— 
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
tf.random.set_seed(1)

# —— Dependencies & Utilities —— 
import sys, argparse, subprocess, json, re
import numpy as np
from pathlib import Path
import pyvista as pv
import xml.etree.ElementTree as ET

try:
    from tqdm.auto import tqdm
    _USE_TQDM = True
except Exception:
    _USE_TQDM = False

# Sionna / Mitsuba
try:
    import sionna
except ImportError:
    os.system("pip install -q sionna")
    import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
import mitsuba as mi

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Sionna:", sionna.__version__)

# ========= Parameters (all relative paths) ==========
# Ray tracing / frequency domain parameters
CENTER_FC = 7e9             # Hz
SUBCARRIER_SPACING = 15e3   # Hz
K = 32                      # Number of subcarriers
MAX_DEPTH = 3               # Maximum interaction order
USE_SYNTHETIC_ARRAY = False # Near-field requires False

# Array definition (receiver UPA, transmitter single antenna)
UPA_VERTICAL = 64
UPA_HORIZONTAL = 64
EL_SPACING = 0.5
AZ_SPACING = 0.5
RX_POLAR = 'V'
TX_POLAR = 'V'

# UAV Z value strategy
RX_USE_TOP_PLUS_OFFSET = False
RX_Z_OFFSET = 0.0
h_receiver =65.0

# ========= Utilities =========
def _read_center_top(ply_path: Path):
    m = pv.read(str(ply_path))
    cx, cy, cz_center = m.center
    z_top = m.bounds[5]
    try: m.clear_data()
    except Exception: pass
    return float(cx), float(cy), float(cz_center), float(z_top)

def _find_uav_ply(time_dir: Path) -> Path | None:
    cand = sorted((time_dir / "mesh").glob("uav_*.ply"))
    if cand:
        return cand[0]
    ue = time_dir / "UE_uav.xml"
    if ue.exists():
        try:
            u = ET.parse(str(ue)).getroot().find(".//uav")
            if u is not None and "id" in u.attrib:
                p = time_dir / "mesh" / f"uav_{u.attrib['id']}.ply"
                if p.exists():
                    return p
        except Exception:
            pass
    cand2 = sorted((time_dir / "mesh").glob("*uav*.ply"))
    if cand2:
        return cand2[0]
    return None

def _sorted_time_dirs(scene_dir: Path):
    times_root = scene_dir / "times"
    tdirs = [d for d in times_root.glob("time_*") if d.is_dir()]
    _re = re.compile(r"_t([\d_]+)$")
    def _key(p: Path):
        m = _re.search(p.name)
        if not m: return 0.0
        return float(m.group(1).replace("_", "."))
    return sorted(tdirs, key=_key)

def _build_frequencies(fc, K, scs):
    k = np.arange(K) - K/ 2.0
    return fc + k * scs

def _rayleigh_distance_from_rx_array(lam,M_V,M_H) -> float:
    D_H = (M_H-1)*lam/2
    D_V = (M_V-1)*lam/2
    D = np.sqrt(D_H**2+D_V**2)
    return 2.0 * (D ** 2) / lam

def _flatten_H_complex(h_complex_tf: tf.Tensor) -> np.ndarray:
    h = h_complex_tf
    if h.ndim != 6:
        raise RuntimeError(f"Unexpected H shape {h.shape}, expect 6-D tensor")
    Nr = h.shape[0] * h.shape[1]
    Nt = h.shape[2] * h.shape[3]
    T  = h.shape[4]
    K  = h.shape[5]
    h4 = h.reshape(Nr, Nt, T, K)
    h3 = h4.reshape(Nr * Nt, T, K)
    h_real = np.real(h3).transpose(1, 0, 2)
    h_imag = np.imag(h3).transpose(1, 0, 2)
    return np.stack([h_real, h_imag], axis=-1)  # [T, M, K, 2]

def _tx_position_from_uav(time_dir: Path) -> np.ndarray | None:
    ply = _find_uav_ply(time_dir)
    if ply is None:
        return None
    cx, cy, z_center, z_top = _read_center_top(ply)
    z = (z_top + RX_Z_OFFSET) if RX_USE_TOP_PLUS_OFFSET else (z_center - 0.6)
    return np.array([cx, cy, z], dtype=float)

def object_array(py_list):
    arr = np.empty((len(py_list),), dtype=object)
    for i, it in enumerate(py_list):
        arr[i] = it
    return arr

def _progress_iter(iterable, desc=""):
    if _USE_TQDM:
        return tqdm(iterable, desc=desc)
    total = len(iterable)
    for i, x in enumerate(iterable, 1):
        if i == 1 or i == total or i % max(1, total // 10) == 0:
            print(f"{desc} {i}/{total}")
        yield x

# ========= Child process: handle and save one subscene =========
def _child_process_one_subscene_cli(main_dir_str: str, sub_dir_str: str, save_root_str: str):
    import os
    from contextlib import contextmanager

    @contextmanager
    def _pushd(path: Path):
        old = os.getcwd()
        os.chdir(str(path))
        try:
            yield
        finally:
            os.chdir(old)

    freqs = _build_frequencies(CENTER_FC, K, SUBCARRIER_SPACING)
    meta = dict(
        center_frequency=CENTER_FC,
        subcarrier_spacing=SUBCARRIER_SPACING,
        num_subcarriers=K,
        rx_array=dict(num_rows=UPA_VERTICAL, num_cols=UPA_HORIZONTAL,
                      v_spacing=EL_SPACING, h_spacing=AZ_SPACING,
                      pattern="tr38901", polarization=RX_POLAR),
        tx_array=dict(num_rows=1, num_cols=1,
                      v_spacing=0.5, h_spacing=0.5,
                      pattern="iso", polarization=TX_POLAR),
        use_synthetic_array=USE_SYNTHETIC_ARRAY,
        max_depth=MAX_DEPTH
    )

    main_dir = Path(main_dir_str)
    sub_dir  = Path(sub_dir_str)
    save_root = Path(save_root_str)
    save_root.mkdir(parents=True, exist_ok=True)

    time_dirs = _sorted_time_dirs(sub_dir)
    H_t_list, pos_t_list, nf_t_list = [], [], []
    sub_stats = {"timepoints_total": len(time_dirs), "timepoints_loaded": 0, "timepoints_skipped": 0}

    rx_array_cfg = dict(
        num_rows=UPA_VERTICAL, num_cols=UPA_HORIZONTAL,
        vertical_spacing=EL_SPACING, horizontal_spacing=AZ_SPACING,
        pattern="tr38901", polarization=RX_POLAR
    )
    tx_array_cfg = dict(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="tr38901", polarization=TX_POLAR
    )
    rx_pos = np.array((0.0, 0.0, float(h_receiver)))

    solver = PathSolver()
    DEBUG_LIMIT = 5
    debug_count = 0

    for t_dir in time_dirs:
        scene_xml = t_dir / "simple_OSM_scene.xml"
        if not scene_xml.exists():
            sub_stats["timepoints_skipped"] += 1
            if debug_count < DEBUG_LIMIT:
                print(f"[SKIP:XML] Missing {scene_xml}")
                debug_count += 1
            continue

        tx_xyz = _tx_position_from_uav(t_dir)
        if tx_xyz is None:
            sub_stats["timepoints_skipped"] += 1
            if debug_count < DEBUG_LIMIT:
                print(f"[SKIP:UAV] Missing UAV-related PLY or UE_uav.xml: {t_dir/'mesh'}")
                debug_count += 1
            continue

        try:
            with _pushd(t_dir):
                try:
                    fr = mi.Thread.thread().file_resolver()
                    fr.append(".")
                    fr.append("mesh")
                except Exception:
                    pass

                scene = load_scene("simple_OSM_scene.xml")
                scene.frequency = CENTER_FC
                scene.rx_array = PlanarArray(**rx_array_cfg)
                scene.tx_array = PlanarArray(**tx_array_cfg)

                rx = Receiver("rx", position=mi.Point3f(float(rx_pos[0]), float(rx_pos[1]), float(rx_pos[2])))
                tx = Transmitter("tx", position=mi.Point3f(float(tx_xyz[0]), float(tx_xyz[1]), float(tx_xyz[2])))
                scene.add(rx); scene.add(tx)
                tx.look_at(rx)

                paths = solver(scene=scene,
                               max_depth=MAX_DEPTH,
                               los=True,
                               specular_reflection=True,
                               diffuse_reflection=False,
                               refraction=True,
                               synthetic_array=USE_SYNTHETIC_ARRAY,
                               seed=42)

            try:
                a_cir, tau = paths.cir()
                num_paths = int(a_cir[0].shape[-2])
            except Exception:
                num_paths = 0

            if num_paths == 0:
                sub_stats["timepoints_skipped"] += 1
                if debug_count < DEBUG_LIMIT:
                    print(f"[SKIP:0PATH] {t_dir} -> Solved successfully but 0 paths found") 
                    debug_count += 1
                try: del scene, paths
                except: pass
                continue

            h_freq = paths.cfr(freqs-CENTER_FC, normalize_delays=False, out_type="numpy")

            H_t = _flatten_H_complex(h_freq)  # [T, M, K, 2]
            if H_t.shape[0] != 1:
                H_t = H_t[:1]
            H_t = H_t[0]

            c0 = 299792458.0
            lam = c0 / CENTER_FC
            R_ray = _rayleigh_distance_from_rx_array(lam, UPA_VERTICAL, UPA_HORIZONTAL)
            dist = float(np.linalg.norm(tx_xyz - np.array((0.0, 0.0, float(h_receiver)))))
            is_nf = 1 if dist < R_ray else 0

            H_t_list.append(H_t[None, ...])
            pos_t_list.append(tx_xyz[None, :])
            nf_t_list.append(np.array([is_nf], dtype=np.int8))
            sub_stats["timepoints_loaded"] += 1

            try: del scene, paths, h_real, h_imag, h_real_tf, h_imag_tf, h_c, H_t
            except: pass

        except Exception as e:
            sub_stats["timepoints_skipped"] += 1
            if debug_count < DEBUG_LIMIT:
                print(f"[SKIP:EXC] {t_dir} -> {type(e).__name__}: {e}")
                debug_count += 1
            continue

    # Save
    if len(H_t_list) > 0:
        H_sub  = np.concatenate(H_t_list, axis=0)
        POS_sub = np.concatenate(pos_t_list, axis=0)
        NF_sub  = np.concatenate(nf_t_list, axis=0)

        out_npz = save_root / f"{main_dir.name}__{sub_dir.name}_channels.npz"
        save_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            H_list=object_array([H_sub]),
            O_T_list=object_array([POS_sub]),
            IS_NEARFIELD_list=object_array([NF_sub]),
            meta=json.dumps(meta, ensure_ascii=False)
        )
        print(f"✓ Subscene saved: {out_npz}")

    out_stats = save_root / f"{main_dir.name}__{sub_dir.name}_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(sub_stats, f, ensure_ascii=False, indent=2)
    print(f"✓ Subscene stats: {out_stats}")

    try: tf.keras.backend.clear_session()
    except: pass
    try:
        import drjit as dr; dr.flush_malloc_cache()
    except: pass
    try: pv.close_all()
    except: pass

# ========= Main process: iterate over "main scene directory collection", save paths mapped by main scene name =========
def build_dataset_batch_for_one_main(main_dir: str | Path, save_base_root: str | Path):
    main_dir = Path(main_dir)
    if not main_dir.exists():
        raise FileNotFoundError(f"main_dir does not exist: {main_dir}")
    save_base_root = Path(save_base_root)
    save_root_for = lambda md: (save_base_root / md.name / "data")

    script_path = sys.argv[0] if sys.argv and sys.argv[0] else __file__

    print(f"\n=== Main scene -> {main_dir} ===")
    sub_scenes = sorted([d for d in main_dir.glob("scene_*") if d.is_dir()])

    scene_stats = {
        "sub_scenes": len(sub_scenes),
        "timepoints_total": 0,
        "timepoints_loaded": 0,
        "timepoints_skipped": 0,
        "subscene_details": {}
    }
    any_saved = False

    for u_idx, sub_dir in enumerate(sub_scenes):
        print(f"  -- Subscene [{u_idx}] -> {sub_dir.name}")

        # Calculate save directory for this main scene: <save_base>/<scene_x>/data
        save_root = save_root_for(main_dir)
        save_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, script_path,
            "--child",
            "--main_dir", str(main_dir),
            "--sub_dir",  str(sub_dir),
            "--save_root", str(save_root),
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  !! Subscene {sub_dir.name} process returned code {ret.returncode}")

        sub_stats_path = save_root / f"{main_dir.name}__{sub_dir.name}_stats.json"
        if sub_stats_path.exists():
            with open(sub_stats_path, "r", encoding="utf-8") as f:
                sub_stats = json.load(f)
            scene_stats["subscene_details"][sub_dir.name] = sub_stats
            scene_stats["timepoints_total"]   += sub_stats.get("timepoints_total", 0)
            scene_stats["timepoints_loaded"]  += sub_stats.get("timepoints_loaded", 0)
            scene_stats["timepoints_skipped"] += sub_stats.get("timepoints_skipped", 0)
            if sub_stats.get("timepoints_loaded", 0) > 0:
                any_saved = True

    if any_saved:
        out_stats_main = save_root_for(main_dir) / f"{main_dir.name}_stats.json"
        with open(out_stats_main, "w", encoding="utf-8") as f:
            json.dump(scene_stats, f, ensure_ascii=False, indent=2)
        print(f"✓ Main scene stats saved: {out_stats_main}")

# Iterate over all main scenes scene_* under the main parent directory, save paths mapped by main scene name
def build_all_main_scenes(main_parent_dir: str | Path, save_base_root: str | Path):
    main_parent_dir = Path(main_parent_dir)
    save_base_root  = Path(save_base_root)
    if not main_parent_dir.exists():
        raise FileNotFoundError(f"Main directory does not exist: {main_parent_dir}")

    main_dirs = sorted([d for d in main_parent_dir.glob("scene_*") if d.is_dir()])
    if not main_dirs:
        raise RuntimeError(f"No scene_* directories found under {main_parent_dir}")
    print(f"Processing {len(main_dirs)} main scenes: {[d.name for d in main_dirs]}")
    for md in main_dirs:
        build_dataset_batch_for_one_main(md, save_base_root)

# ========= CLI entry =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterate over each main scene scene_* under the main directory, generate data for each subscene; output mapped by main scene name to .../<scene_id>/data/"
    )
    parser.add_argument("--child", action="store_true", help="Child process mode: only process one subscene")
    parser.add_argument("--main_parent_dir", type=str,
                        default="DownstreamDataset/OriginalOSM_Val/Urban/Nanjing",
                        help="Main directory (contains multiple main scenes scene_*)")
    parser.add_argument("--main_dir", type=str, default="", help="(For child process) Main scene directory")
    parser.add_argument("--sub_dir",  type=str, default="", help="(For child process) Subscene directory")
    parser.add_argument("--save_base_root", type=str,
                        default="DownstreamDataset/Hybrid_Channel_UPA64X64/ValSet/Urban/Nanjing",
                        help="Base output directory (final output will be written to <save_base_root>/<scene_id>/data/)")
    parser.add_argument("--save_root", type=str, default="", help="(For child process) Final save directory")
    args = parser.parse_args()

    if args.child:
        if not args.main_dir or not args.sub_dir or not args.save_root:
            raise SystemExit("Child process missing parameters: --main_dir / --sub_dir / --save_root")
        _child_process_one_subscene_cli(args.main_dir, args.sub_dir, args.save_root)
        sys.exit(0)

    # Top-level: iterate over each main scene scene_*
    build_all_main_scenes(args.main_parent_dir, args.save_base_root)
