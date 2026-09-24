"""Check that report-width figure crops do not inherit full-figure notes."""

import json
from pathlib import Path

import pytest

from scripts import plot_diagnostic_confirmation as plot


RESULTS = Path(__file__).resolve().parents[1] / "results/confirmation"


@pytest.mark.skipif(
    any(
        not (RESULTS / name).is_file()
        for name in (
            "diagnostic_baseline_confirmation_comparison.json",
            "diagnostic_baseline_confirmation_capable_control.json",
        )
    ),
    reason="confirmation results are not distributed with the code",
)
def test_panel_crops_exclude_full_figure_note(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    comparison = json.loads(
        (root / "results/confirmation/diagnostic_baseline_confirmation_comparison.json").read_text()
    )
    capable = json.loads(
        (root / "results/confirmation/diagnostic_baseline_confirmation_capable_control.json").read_text()
    )
    captured = {}

    def record_full(fig, _path):
        captured["full"] = [item.get_text() for item in fig.texts]

    def record_split(fig, _path):
        captured["split"] = [item.get_text() for item in fig.texts]

    monkeypatch.setattr(plot, "save_figure", record_full)
    monkeypatch.setattr(plot, "save_split_figure", record_split)
    plot.render_result_figure(comparison, capable, tmp_path / "figure")

    assert any("Primary endpoint:" in text for text in captured["full"])
    assert not any("Primary endpoint:" in text for text in captured["split"])
