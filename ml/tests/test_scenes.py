import json

import numpy as np
import pytest

from rfea_ml.scenes import (
    DECIMATION_FRACTION,
    N_CHANNELS,
    SCENE_FS,
    SceneConfig,
    generate,
    make_burst,
    schedule,
)

SHORT = SceneConfig(name="t", duration_s=1.0, burst_s=(0.2, 0.4), gap_s=(0.02, 0.1), seed=3)


def test_schedule_never_overlaps_in_frequency() -> None:
    bursts = schedule(SceneConfig(duration_s=60, seed=1), np.random.default_rng(1))
    assert len(bursts) > 20
    for i, a in enumerate(bursts):
        assert 0 <= a["channel"] < N_CHANNELS
        for b in bursts[i + 1 :]:
            if a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]:
                assert a["channel"] != b["channel"]


def test_burst_power_follows_classifier_snr() -> None:
    b = {"start_s": 0.0, "end_s": 0.3, "channel": 2, "label": "QPSK", "tsig_class": "qpsk"}
    x, ann = make_burst(b, SHORT, np.random.default_rng(0))
    assert len(x) == ann["core:sample_count"] == int(0.3 * SCENE_FS)
    ramp = int(1e-3 * SCENE_FS)
    p = np.mean(np.abs(x[ramp:-ramp]) ** 2)
    f_dec = ann["rfea:measured_bandwidth_hz"] / DECIMATION_FRACTION
    expected = 10 ** (ann["rfea:snr_db"] / 10) * f_dec / SCENE_FS
    assert p == pytest.approx(expected, rel=0.01)
    lo, hi = SHORT.bandwidth_hz
    assert 0.5 * lo < ann["rfea:measured_bandwidth_hz"] < 2 * hi


def test_generate_writes_sigmf(tmp_path) -> None:
    meta_path = generate(SHORT, tmp_path)
    meta = json.loads(meta_path.read_text())
    x = np.fromfile(meta_path.with_suffix(".sigmf-data"), dtype="<c8")
    assert len(x) == int(SHORT.duration_s * SCENE_FS)
    assert meta["global"]["core:datatype"] == "cf32_le"
    fc = meta["captures"][0]["core:frequency"]
    anns = meta["annotations"]
    assert anns
    starts = [a["core:sample_start"] for a in anns]
    assert starts == sorted(starts)
    for a in anns:
        assert a["core:sample_start"] + a["core:sample_count"] <= len(x)
        assert fc - SCENE_FS / 2 <= a["core:freq_lower_edge"] < a["core:freq_upper_edge"]
        assert a["core:freq_upper_edge"] <= fc + SCENE_FS / 2
    with pytest.raises(FileExistsError):
        generate(SHORT, tmp_path)
