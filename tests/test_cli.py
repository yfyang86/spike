"""Tests for the unified CLI and its vector/raster/scanned-PDF routing."""
import os

import pytest

pytest.importorskip("scipy")
pytest.importorskip("PIL")

from spike import cli  # noqa: E402


# -- input-type detection ------------------------------------------------- #
def test_count_path_ops_distinguishes_vector_from_scanned():
    vector = "0 0 m " + "1 1 l " * 500          # a drawn waveform
    scanned = "q 600 0 0 800 0 0 cm /Im0 Do Q"  # an image-only page
    assert cli._count_path_ops(vector) >= cli.PDF_VECTOR_MIN_OPS
    assert cli._count_path_ops(scanned) == 0


def test_resolve_kind_image_extensions():
    assert cli.resolve_kind("scan.png", "auto") == "raster"
    assert cli.resolve_kind("photo.JPG", "auto") == "raster"
    assert cli.resolve_kind("scan.png", "raster") == "raster"


def test_resolve_kind_forced_modes_on_pdf(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    assert cli.resolve_kind(str(p), "vector") == "vector"
    assert cli.resolve_kind(str(p), "raster") == "raster-pdf"


def test_resolve_kind_auto_routes_scanned_pdf_to_raster(tmp_path):
    # a PDF with no readable vector content stream -> scanned -> raster-pdf
    p = tmp_path / "scanned.pdf"
    p.write_bytes(b"%PDF-1.4\n% image-only, no path operators\n")
    assert cli.resolve_kind(str(p), "auto") == "raster-pdf"


def test_resolve_kind_auto_keeps_vector_pdf(tmp_path, monkeypatch):
    p = tmp_path / "vec.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(cli, "_pdf_is_vector", lambda path: True)
    assert cli.resolve_kind(str(p), "auto") == "vector"


# -- end-to-end through the CLI ------------------------------------------- #
def test_cli_raster_roundtrip_writes_outputs(tmp_path, capsys):
    from synth import synthetic_ecg_png

    img = tmp_path / "synth.png"
    synthetic_ecg_png(str(img))
    out = tmp_path / "out"
    cli.main([str(img), "-o", str(out), "--type", "raster"])

    base = str(out / "synth")
    for suffix in ("_rows.csv", "_leads.csv", "_meta.json"):
        assert os.path.getsize(base + suffix) > 0
    assert "raster image" in capsys.readouterr().out


def test_cli_missing_input_errors():
    with pytest.raises(SystemExit):
        cli.main(["does_not_exist.png"])


def test_cli_auto_conflicts_with_explicit_type(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n")
    with pytest.raises(SystemExit):
        cli.main([str(img), "--auto", "--type", "vector"])


def test_count_path_ops_ignores_lone_letters_in_text():
    # isolated 'l'/'c' inside drawn text must NOT count as path operators
    assert cli._count_path_ops("BT (a l c v m b) Tj ET") == 0
