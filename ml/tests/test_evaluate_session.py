from rfea_ml.evaluate_session import evaluate, iou_f, match


def ref(start, end, f_lo, f_hi, label="BPSK", snr=12.0):
    return {"start_s": start, "end_s": end, "f_lo": f_lo, "f_hi": f_hi, "label": label,
            "known": label in ("BPSK", "QPSK"), "snr_db": snr}  # fmt: skip


def event(t, center, bw, status="identified", label="BPSK", frames=2):
    return {"track_id": int(t * 100), "detected_at_s": t, "decided_at_s": t + 0.1,
            "center_freq_hz": center, "bandwidth_hz": bw, "status": status, "label": label,
            "confidence": 0.99, "n_frames": frames, "observation_s": 0.1}  # fmt: skip


def test_iou_f() -> None:
    a = ref(0, 1, 100, 200)
    assert iou_f(event(0, 150, 100), a) == 1
    assert iou_f(event(0, 200, 100), a) == 50 / 150


def test_match_needs_time_and_frequency_and_is_one_to_one() -> None:
    refs = [ref(1.0, 3.0, 100, 200), ref(5.0, 6.0, 100, 200)]
    events = [
        event(1.0, 150, 100),  # signal 0
        event(1.2, 150, 100),  # second event of the same signal: false
        event(5.5, 400, 100),  # right time, wrong frequency: false
        event(4.0, 150, 100),  # right frequency, no signal on the air: false
    ]
    assert match(events, refs) == {0: 0}


def test_evaluate_counts_outcomes() -> None:
    refs = [ref(0, 2, 100, 200), ref(0, 2, 300, 400, "QPSK", snr=-2.0),
            ref(0, 2, 500, 600, "ofdm-64"), ref(3, 4, 700, 800)]  # fmt: skip
    events = [event(0.1, 150, 100), event(0.1, 350, 100, "unknown", None, 20),
              event(0.1, 550, 100, "identified", "AM-SSB"), event(1.0, 900, 50)]  # fmt: skip
    r = evaluate(events, refs)
    d = r["detection"]
    assert (d["matched"], d["false_events"], d["recall"]) == (3, 1, 0.75)
    c = r["classification"]["all"]
    assert (c["signals"], c["detected"], c["correct"], c["unknown"]) == (3, 2, 0.5, 0.5)
    assert c["decided_accuracy"] == 1.0
    assert r["classification"]["< 0"]["unknown"] == 1.0
    assert r["outside_training"] == {"signals": 1, "detected": 1, "unknown": 0.0,
                                     "called": {"AM-SSB": 1}}  # fmt: skip
