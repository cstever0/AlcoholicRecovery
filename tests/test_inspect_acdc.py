"""Tests for the ACDC dataset inspection and validation script."""

from pathlib import Path
import pytest

from crip.mri.inspect_acdc import (
    inspect_acdc_dataset,
    parse_info_cfg,
    validate_patient,
)


def test_parse_info_cfg_valid(tmp_path: Path):
    cfg_file = tmp_path / "Info.cfg"
    cfg_file.write_text(
        "ED: 1\nES: 10\nGroup: NOR\nHeight: 175.5\nWeight: 72.0\nNbFrame: 25\n",
        encoding="utf-8",
    )
    meta, errors = parse_info_cfg(cfg_file, "patient001")
    assert not errors
    assert meta is not None
    assert meta.ed_frame == 1
    assert meta.es_frame == 10
    assert meta.group == "NOR"
    assert meta.height == 175.5
    assert meta.weight == 72.0
    assert meta.nb_frame == 25


def test_parse_info_cfg_missing(tmp_path: Path):
    cfg_file = tmp_path / "nonexistent.cfg"
    meta, errors = parse_info_cfg(cfg_file, "patient001")
    assert meta is None
    assert any("not found" in e for e in errors)


def test_parse_info_cfg_invalid_numbers(tmp_path: Path):
    cfg_file = tmp_path / "Info.cfg"
    cfg_file.write_text("ED: -1\nES: 0\n", encoding="utf-8")
    meta, errors = parse_info_cfg(cfg_file, "patient001")
    assert any("Invalid ED" in e for e in errors)


def test_inspect_acdc_dataset_runs_on_synthetic(synthetic_acdc_dir: Path, tmp_path: Path):
    out_json = tmp_path / "report.json"
    report = inspect_acdc_dataset(synthetic_acdc_dir, output_report_path=out_json)
    assert out_json.exists()
    assert report["summary"]["total_patient_directories"] == 1
    assert report["summary"]["successfully_validated_patients"] == 1
    assert report["summary"]["validation_status"] == "PASS"


def test_inspect_acdc_missing_dataset_raises(tmp_path: Path):
    nonexistent = tmp_path / "nonexistent_dir"
    with pytest.raises(FileNotFoundError):
        inspect_acdc_dataset(nonexistent)
