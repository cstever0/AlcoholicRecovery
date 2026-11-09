"""Fill design/src/data/real_cases.json with measured volumes and images from the ACDC scans.

Run from the repo root, with .venv active, on a machine that has the ACDC training data:

    python design/export_real_cases.py [--acdc-dir data/raw/acdc/training]

For CRIP-DEMO-CASE-A/B/C (ACDC patient001-003) it:
  1. computes LV/RV volumes, LVEF, RVEF and LV mass from the ACDC expert masks
     (the same compute_biventricular_measurements call that
     crip.integration.workflow uses for use_automated_mri=False),
  2. saves one mid-ventricular end-diastole slice and its contour overlay per case
     as small JPGs in design/src/data/,
  3. rewrites real_cases.json with those values, provenance REAL_MRI.

It never falls back to default values. If any scan, mask or measurement is missing or
implausible it stops with an error and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from crip.integration.demo_cases import DEMO_SCENARIO_CONFIGS
from crip.mri.acdc_loader import (
    DEFAULT_ACDC_TRAINING_DIR,
    ACDCDataError,
    ACDCDataLoader,
    normalize_for_display,
)
from crip.mri.cardiac_metrics import compute_biventricular_measurements

DATA_DIR = Path(__file__).resolve().parent / "src" / "data"
CASES_FILE = DATA_DIR / "real_cases.json"
IMAGE_SIZE = 256
# Same colors as --mri-rv, --mri-myo and --mri-lv in design/src/styles.css.
LABEL_COLORS = {1: (0x26, 0xA6, 0xFF), 2: (0x33, 0xD9, 0x59), 3: (0xFF, 0x40, 0x40)}
OVERLAY_ALPHA = 0.45


class ExportError(RuntimeError):
    pass


def measure(loader: ACDCDataLoader, acdc_id: str):
    if acdc_id not in loader.list_patients():
        raise ExportError(f"{acdc_id} not found in {loader.dataset_root.resolve()}")
    ed = loader.load_cardiac_frame(acdc_id, phase="ED", load_mask=True)
    es = loader.load_cardiac_frame(acdc_id, phase="ES", load_mask=True)
    if ed.mask_data is None or es.mask_data is None:
        raise ExportError(f"{acdc_id}: expert ED/ES masks are missing")

    m = compute_biventricular_measurements(
        ed_mask=ed.mask_data, es_mask=es.mask_data, voxel_spacing_mm=ed.voxel_spacing, patient_id=acdc_id,
    )
    if not m.is_physiologically_valid:
        raise ExportError(f"{acdc_id}: implausible measurements: {m.validation_warnings}")

    r1 = lambda v: round(float(v), 1)  # noqa: E731
    values = {
        "baseline_lvef": r1(m.lvef_percent),
        "edv_ml": r1(m.lvedv_ml),
        "esv_ml": r1(m.lvesv_ml),
        "stroke_volume_ml": r1(m.lvsv_ml),
        "rvedv_ml": r1(m.rvedv_ml),
        "rvesv_ml": r1(m.rvesv_ml),
        "rvef_percent": r1(m.rvef_percent),
        "myocardial_mass_g": r1(m.myocardial_mass_ed_g),
    }
    for key, v in values.items():
        if not np.isfinite(v) or v <= 0:
            raise ExportError(f"{acdc_id}: {key} = {v} is not a usable measurement")
    return values, ed


def slice_images(frame, acdc_id: str) -> tuple[Image.Image, Image.Image, int, int]:
    with_lv = [i for i in range(frame.mask_data.shape[0]) if (frame.mask_data[i] == 3).any()]
    if not with_lv:
        raise ExportError(f"{acdc_id}: no LV cavity in the end-diastole mask")
    idx = with_lv[len(with_lv) // 2]

    gray = (normalize_for_display(frame.image_data[idx]) * 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    overlay = rgb.copy()
    mask = frame.mask_data[idx]
    for label, color in LABEL_COLORS.items():
        sel = mask == label
        overlay[sel] = (1 - OVERLAY_ALPHA) * overlay[sel] + OVERLAY_ALPHA * np.array(color, np.float32)

    def to_image(arr: np.ndarray) -> Image.Image:
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
        scale = IMAGE_SIZE / max(img.size)
        return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

    return to_image(rgb), to_image(overlay), idx + 1, int(frame.mask_data.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--acdc-dir", type=Path, default=DEFAULT_ACDC_TRAINING_DIR,
                        help=f"ACDC training folder (default: {DEFAULT_ACDC_TRAINING_DIR})")
    args = parser.parse_args()

    loader = ACDCDataLoader(dataset_root=args.acdc_dir)
    if not loader.is_available():
        raise ExportError(
            f"No ACDC scans at {args.acdc_dir.resolve()}. Run from the repo root on a machine with the "
            "ACDC training data, or pass --acdc-dir. Nothing was written."
        )

    doc = json.loads(CASES_FILE.read_text())
    by_id = {rec["scenario_id"]: rec for rec in doc["records"]}
    missing = set(DEMO_SCENARIO_CONFIGS) - set(by_id)
    if missing:
        raise ExportError(f"real_cases.json has no record for {sorted(missing)}")

    # Measure everything first so a failure on any case leaves the files untouched.
    results = {}
    for case_id, cfg in DEMO_SCENARIO_CONFIGS.items():
        acdc_id = cfg["acdc_patient_id"]
        if by_id[case_id]["acdc_id"] != acdc_id:
            raise ExportError(f"{case_id}: real_cases.json says {by_id[case_id]['acdc_id']}, demo_cases.py says {acdc_id}")
        values, ed_frame = measure(loader, acdc_id)
        results[case_id] = (acdc_id, values, slice_images(ed_frame, acdc_id))

    for case_id, (acdc_id, values, (original, overlay, slice_no, slices)) in results.items():
        rec = by_id[case_id]
        names = {k: f"acdc-{acdc_id}-ed-{k}.jpg" for k in ("original", "overlay")}
        original.save(DATA_DIR / names["original"], quality=85)
        overlay.save(DATA_DIR / names["overlay"], quality=85)
        rec.update(values)
        rec["lvef_method"] = "EXPERT_MASK"
        rec["image"] = {**names, "phase": "ED", "slice": slice_no, "slices": slices}
        for key in values:
            rec["provenance"][key] = "REAL_MRI"
        print(f"{case_id} ({acdc_id}): LVEF {values['baseline_lvef']}%, LV EDV {values['edv_ml']} mL, "
              f"RVEF {values['rvef_percent']}%, LV mass {values['myocardial_mass_g']} g, ED slice {slice_no}/{slices}")

    doc["notice"] = doc["notice"].split(" Volumes are PENDING_EXPORT")[0] + (
        " Volumes, LVEF and images come from the ACDC expert masks via design/export_real_cases.py."
    )
    CASES_FILE.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {CASES_FILE}. Now run `npm run build` in design/.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ExportError, ACDCDataError) as e:
        print(f"export_real_cases: ERROR: {e}", file=sys.stderr)
        sys.exit(1)
