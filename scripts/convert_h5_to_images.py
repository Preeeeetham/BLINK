"""
HDF5 Satellite Data to JPEG / PNG Converter & Visualizer for Project BLINK.

Reads .h5 / .hdf5 / .nc satellite datasets and exports:
1. Grayscale & Colormapped Thermal Infrared (IMG_TIR1, IMG_TIR2)
2. Water Vapour (IMG_WV)
3. Visible (IMG_VIS) & Short-Wave IR (IMG_SWIR)
4. Multi-Spectral False-Color RGB Composite
"""

import os
import sys
from pathlib import Path
import glob
import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def convert_h5_file(h5_path: str, output_dir: str = "data/exported_images", max_dim: int = 2816):
    """Converts a single valid HDF5 satellite file into JPG/PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(h5_path).stem
    print(f"\n[CONVERT] Processing: {h5_path}")

    try:
        with h5py.File(h5_path, 'r') as h5f:
            datasets = {}
            for k in h5f.keys():
                if isinstance(h5f[k], h5py.Dataset):
                    datasets[k] = h5f[k]

            print(f"  Found datasets: {list(datasets.keys())}")
            if not datasets:
                print("  [WARN] No datasets found inside HDF5.")
                return []

            exported_files = []
            normalized_channels = {}

            # Primary meteorological channels
            target_channels = ['IMG_TIR1', 'IMG_TIR2', 'IMG_WV', 'IMG_VIS', 'IMG_MIR', 'IMG_SWIR']

            for name in target_channels:
                if name not in datasets:
                    continue
                ds = datasets[name]
                arr = ds[()]
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.ndim != 2:
                    continue

                # Downsample if exceedingly large (e.g. 11264x11220 VIS band -> 2816x2805)
                h, w = arr.shape
                if h > max_dim or w > max_dim:
                    step_y = int(np.ceil(h / max_dim))
                    step_x = int(np.ceil(w / max_dim))
                    arr = arr[::step_y, ::step_x]

                clean_name = name.replace("/", "_")
                
                # Robust min-max percentile scaling
                valid_mask = np.isfinite(arr) & (arr > 0)
                if np.any(valid_mask):
                    vmin, vmax = np.percentile(arr[valid_mask], (1, 99))
                else:
                    vmin, vmax = float(arr.min()), float(arr.max())
                
                if vmax <= vmin:
                    vmax = vmin + 1.0

                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
                normalized_channels[name] = norm

                # 1. Grayscale JPG
                gray_img = Image.fromarray((norm * 255).astype(np.uint8))
                gray_path = os.path.join(output_dir, f"{basename}_{clean_name}_gray.jpg")
                gray_img.save(gray_path, quality=90)
                exported_files.append(gray_path)
                print(f"  -> Exported: {gray_path} ({gray_img.size[0]}x{gray_img.size[1]})")

                # 2. Colored Colormap JPG
                if "TIR" in name or "MIR" in name:
                    cmap = cm.magma
                elif "WV" in name:
                    cmap = cm.viridis
                else:
                    cmap = cm.bone

                colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
                color_img = Image.fromarray(colored)
                color_path = os.path.join(output_dir, f"{basename}_{clean_name}_colored.jpg")
                color_img.save(color_path, quality=90)
                exported_files.append(color_path)
                print(f"  -> Exported: {color_path}")

            # 3. Export False Color RGB Composite (VIS + WV + TIR1)
            if 'IMG_TIR1' in normalized_channels and 'IMG_WV' in normalized_channels:
                tir1_norm = normalized_channels['IMG_TIR1']
                wv_norm = normalized_channels['IMG_WV']
                h_target, w_target = tir1_norm.shape

                # Resize VIS or MIR to match target resolution
                if 'IMG_VIS' in normalized_channels:
                    vis_img = Image.fromarray((normalized_channels['IMG_VIS'] * 255).astype(np.uint8))
                    vis_norm = np.array(vis_img.resize((w_target, h_target), Image.Resampling.BILINEAR)) / 255.0
                else:
                    vis_norm = tir1_norm

                rgb = np.stack([vis_norm, wv_norm, tir1_norm], axis=-1)
                rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))
                rgb_path = os.path.join(output_dir, f"{basename}_RGB_COMPOSITE.jpg")
                rgb_img.save(rgb_path, quality=92)
                exported_files.append(rgb_path)
                print(f"  -> Exported RGB Composite: {rgb_path}")

            return exported_files

    except Exception as e:
        print(f"  [ERROR] Could not convert {h5_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def convert_all_h5_in_directory(search_dir: str, output_dir: str = "data/exported_images"):
    """Scans search_dir for all .h5 files and converts valid ones."""
    files = sorted(glob.glob(os.path.join(search_dir, "*.h5")))
    print(f"Found {len(files)} .h5 files in {search_dir}")
    all_exports = []
    for f in files:
        exports = convert_h5_file(f, output_dir)
        all_exports.extend(exports)
    return all_exports


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "MOSDAC"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/exported_images"
    convert_all_h5_in_directory(target, out)
