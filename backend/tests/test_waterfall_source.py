import numpy as np
import pytest
from conftest import FS, tone

from rfea_backend.dsp.waterfall import Waterfall, WaterfallConfig, row_power_db
from rfea_backend.sources.virtual import VirtualSource, list_scenes


def test_tone_lands_in_its_bin() -> None:
    p = row_power_db(tone(100e3, 0.05), 1024)
    expected = 512 + round(100e3 / FS * 1024)
    assert abs(int(np.argmax(p)) - expected) <= 1


def test_quantization_is_relative_to_noise() -> None:
    wf = Waterfall(WaterfallConfig())
    rng = np.random.default_rng(0)
    noise = ((rng.normal(size=50_000) + 1j * rng.normal(size=50_000)) / np.sqrt(2)).astype(
        np.complex64
    )
    p, q, lo, hi = wf.row(noise + tone(-200e3, 0.05) * 100)
    assert hi - lo == pytest.approx(60)
    assert lo == pytest.approx(np.median(p) - 10)
    # noise sits at 10 dB above lo: 255 * 10 / 60 ~ 42; the tone saturates
    assert abs(int(np.median(q)) - 42) <= 3
    assert q.max() == 255
    assert q.dtype == np.uint8


def test_virtual_source_reads_blocks_to_the_end(scenes_dir) -> None:
    [info] = list_scenes(scenes_dir)
    assert info.name == "tone"
    assert info.duration_s == pytest.approx(0.3)
    src = VirtualSource(scenes_dir, "tone")
    sizes = []
    while (b := src.read(50_000)) is not None:
        sizes.append(len(b))
    assert sizes == [50_000] * 6
    assert src.position == 300_000
    src.close()


@pytest.mark.parametrize("name", ["missing", "../tone", ""])
def test_virtual_source_rejects_unknown_names(scenes_dir, name: str) -> None:
    with pytest.raises(FileNotFoundError):
        VirtualSource(scenes_dir, name)
