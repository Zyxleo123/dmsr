"""Reporting-layer regression tests for scripts/dmsr_eval.py.

These guard a class of bug that is invisible in the numbers themselves: metrics
that are computed correctly per-crop but silently *dropped* on the way into the
summary and the decision rule.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def de():
    spec = importlib.util.spec_from_file_location("dmsr_eval", REPO / "scripts" / "dmsr_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    """Baseline rows FIRST and with fewer keys -- the real file layout."""
    base = [{"model": "baseline_upsample", "crop": i,
             "rk_transition": 0.87, "rk_low": 0.999} for i in range(3)]
    flow = [{"model": "stage_a", "crop": i,
             "rk_transition": 0.89, "rk_low": 0.999,
             "bispectrum_error": 0.5 + 0.01 * i,
             "squeezed_cross_bispectrum_error": 0.7,
             "sample_diversity": 0.55} for i in range(3)]
    return base + flow


def test_summary_keeps_metrics_the_first_model_does_not_emit(de):
    """REGRESSION: key set must be a union over all rows, not rows[0].

    The A_plus(y) baseline is listed first and skips the bispectra. Taking keys
    from rows[0] silently deleted bispectrum_error and
    squeezed_cross_bispectrum_error -- 2 of 4 pre-registered primary metrics and
    1 of 2 conditional ones -- from summary.json and the printed table.
    """
    s = de._summarise(_rows())
    assert np.isfinite(s["stage_a"]["bispectrum_error"])
    assert np.isfinite(s["stage_a"]["squeezed_cross_bispectrum_error"])
    assert s["stage_a"]["bispectrum_error"] == pytest.approx(0.51)
    # The baseline genuinely lacks them: NaN is correct, absence is not.
    assert np.isnan(s["baseline_upsample"]["bispectrum_error"])


def test_metric_keys_is_a_union_in_first_seen_order(de):
    keys = de._metric_keys(_rows())
    for k in ("rk_transition", "rk_low", "bispectrum_error",
              "squeezed_cross_bispectrum_error", "sample_diversity"):
        assert k in keys
    assert "model" not in keys and "crop" not in keys


def test_all_pre_registered_metrics_are_reachable(de):
    """Every metric the decision rule consults must be produced by evaluate_batch.

    A typo in PRIMARY_METRICS/GUARD_METRICS would make decision_rule silently skip
    that metric (it drops non-finite entries), quietly weakening the C-vs-D test.
    """
    from cosmo_sr.dmsr import evaluate as ev
    import inspect

    src = inspect.getsource(ev)
    for name in de.PRIMARY_METRICS + de.CONDITIONAL_METRICS + de.GUARD_METRICS:
        base = name.replace("rk_", "").replace("Tk_error_", "")
        assert f'"{name}"' in src or f"{base}" in src, (
            f"{name} is in the decision rule but never produced by evaluate.py"
        )


def test_decision_rule_flags_a_reduced_metric_set(de):
    """A NaN primary metric must not silently count as 'no improvement'."""
    stats = {
        "rk_transition": {"rel_improvement": 0.10, "lower_is_better": False},
        "squeezed_cross_bispectrum_error": {"rel_improvement": float("nan"),
                                            "lower_is_better": True},
    }
    v = de.decision_rule(stats)
    assert "rk_transition" in v["improved_metrics"]
    assert "squeezed_cross_bispectrum_error" not in v["improved_metrics"]
    # Only one improvement survives -> must NOT be declared a core success.
    assert v["verdict"] != "core success"
