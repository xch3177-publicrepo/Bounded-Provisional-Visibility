import asyncio, sys, json
sys.path.insert(0, "PUBLIC_HOME/projects/paper-venue-scout/prototype")
import standalone_experiments as S

S.QUIESCE.update(calibration_reads=5, calibration_gap_s=0, cooldown_s=0,
                 probe_gap_s=0, consecutive_required=2, max_wait_s=999, max_attempts=6)
seq = []
async def fake_sentinel(b, qv, n=200):
    v = seq.pop(0)
    return {"sentinel_p50": v/2, "sentinel_p95": v, "sentinel_n": 200}
S.s1_sentinel = fake_sentinel

async def main():
    # calibration: five tight reads -> band from MAD with the rel_tol floor
    seq[:] = [0.0020, 0.0021, 0.0020, 0.0022, 0.0020]
    cal = await S.s1_calibrate(None, None)
    assert abs(cal["median_p95"] - 0.0020) < 1e-9
    assert cal["band_lo"] < 0.0020 < cal["band_hi"], cal
    print("  band %.2f-%.2f ms" % (1000*cal["band_lo"], 1000*cal["band_hi"]))

    # out, out, in, in -> passes on the fourth probe, and the reading handed to
    # the block is the last in-band one
    seq[:] = [0.0064, 0.0050, 0.0021, 0.0020]
    last, log = await S.s1_quiesce(None, None, 1, cal)
    assert log["passed"] and len(log["attempts"]) == 4, log
    assert last["sentinel_p95"] == 0.0020
    assert [a["in_band"] for a in log["attempts"]] == [False, False, True, True]
    print("  recovers after 2 out-of-band probes: passed, attempts=4")

    # in, out, in, in -> the run counter resets, so it must NOT pass at probe 3
    seq[:] = [0.0020, 0.0064, 0.0021, 0.0020]
    last, log = await S.s1_quiesce(None, None, 2, cal)
    assert len(log["attempts"]) == 4, log
    print("  a single in-band reading does not count: consecutive rule holds")

    # never returns -> aborts the whole run rather than proceeding
    seq[:] = [0.0064]*40   # never in band -> must hit the probe cap
    try:
        await S.s1_quiesce(None, None, 3, cal)
        raise AssertionError("should have aborted")
    except SystemExit as e:
        assert "probe cap" in str(e), str(e)
        print("  never settles: SystemExit at the probe cap (no skip, no widen)")
    print("\n==== 4/4 quiesce checks passed ====")

asyncio.run(main())
