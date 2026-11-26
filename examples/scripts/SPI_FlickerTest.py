# -*- coding: utf-8 -*-
"""
Flicker diagnosis for the HTPA160x120dR1 over SPI.

Records raw frames and, per frame, the mean digits of every block (top and
bottom half), the time since the previous conversion START before block 0
and whether the frame read fresh electrical offsets. Then prints how much
each block fluctuates from frame to frame and whether block 0 depends on
the gap or on the eloff phase.

Block x of the top and the bottom half are converted together (Section 8);
block 0 holds the top 5 rows of the top half and the bottom 5 rows of the
bottom half.

Cover the lens (or look at a uniform, stable scene) while recording.


Usage Examples
==============

On the Pi (records and analyses):
python ./SPI_FlickerTest.py --bus 0 --frames 300 --eloff_every 8 --gap_ms 0
python ./SPI_FlickerTest.py --bus 0 --frames 300 --eloff_every 8 --gap_ms 50
python ./SPI_FlickerTest.py --bus 0 --frames 300 --eloff_every 1 --gap_ms 50

Analyse a recording again, e.g. on the PC:
python ./SPI_FlickerTest.py --analyze flicker_e8_g50.npz


Command-line Arguments
======================
--bus : int
    SPI bus that the HTPA device is connected to.
--frames : int (default 300)
    Number of frames to record.
--eloff_every : int (default 8)
    Passed to the driver: fresh electrical offsets every n-th frame.
--gap_ms : float (default 0)
    Extra pause between frames, drawn uniformly from [0, gap_ms]. Imitates
    the irregular idle time the threaded pipeline causes before block 0.
--out : str (optional)
    File the recording is saved to (default flicker_e<n>_g<ms>.npz).
--analyze : str (optional)
    Skip recording, only analyse this .npz file.
"""

import argparse
import random
import time

import numpy as np

_BLOCKS = 12
_ROWS, _COLS = 120, 160
_ROWS_BLOCK = 5

# Block index of every image row: top half top-down, bottom half bottom-up
_ROW_BLOCK = np.array([r // _ROWS_BLOCK if r < _ROWS // 2
                       else (_ROWS - 1 - r) // _ROWS_BLOCK
                       for r in range(_ROWS)])
_ROW_HALF = (np.arange(_ROWS) >= _ROWS // 2).astype(int)


def block_means(img: np.ndarray) -> np.ndarray:
    """(N, 19200) frames -> (N, 2, 12) mean per half and block."""
    rows = img.reshape(len(img), _ROWS, _COLS).mean(axis=2)      # (N, 120)
    out = np.empty((len(img), 2, _BLOCKS))
    for h in range(2):
        for b in range(_BLOCKS):
            out[:, h, b] = rows[:, (_ROW_HALF == h) & (_ROW_BLOCK == b)].mean(axis=1)
    return out


def make_ee_enable(pin: int = 8, chip: int = 0):
    import lgpio
    h = lgpio.gpiochip_open(chip)
    lgpio.gpio_claim_output(h, pin, 1)           # idle HIGH
    write = lgpio.gpio_write

    def ee_enable(level: bool) -> None:
        write(h, pin, level)
    ee_enable.handle = h
    return ee_enable


def record(d, n_frames: int, gap_ms: float) -> dict:
    """Record n_frames raw frames with driver d, see module docstring."""
    from hspy.drivers.spi.drivers import _CMD_CONFIG, _BIT_START, _BIT_BLIND

    # Time stamp every conversion START: (perf_counter, BLIND bit set)
    starts = []
    write_register = d._write_register

    def logged_write_register(cmd, value):
        if cmd == _CMD_CONFIG and value & _BIT_START:
            starts.append((time.perf_counter(), bool(value & _BIT_BLIND)))
        write_register(cmd, value)

    d._write_register = logged_write_register

    pix, eloff, ptat, vdd = [], [], [], []
    gap, dt_blocks, phase = [], [], []
    try:
        for i in range(n_frames):
            first = len(starts)
            raw = d.acquire_raw()
            phase.append(d._n_since_eloff)       # 1 = fresh eloff this frame
            data = d.convert_spi_data(raw)

            pix.append(data['pix'].copy())
            eloff.append(data['eloff'].copy())
            ptat.append(data['ptat'].copy())
            vdd.append(data['vdd'].copy())

            # read_frame() runs before the BLIND conversion, so the first
            # START of this frame is block 0
            t_block = [t for t, blind in starts[first:] if not blind]
            gap.append(t_block[0] - starts[first - 1][0] if first else np.nan)
            dt_blocks.append(np.median(np.diff(t_block)))

            if gap_ms > 0:
                time.sleep(random.uniform(0, gap_ms) * 1e-3)
    finally:
        d._write_register = write_register

    return dict(pix=np.array(pix), eloff=np.array(eloff),
                ptat=np.array(ptat), vdd=np.array(vdd),
                gap_ms=np.array(gap) * 1e3, dt_blocks_ms=np.array(dt_blocks) * 1e3,
                phase=np.array(phase), eloff_every=d.eloff_every, gap_max_ms=gap_ms)


def analyze(r) -> None:
    # Frame 0 is the first one after init() and has no gap: left out
    pix = r['pix'][1:].astype(np.float64)
    comp = pix - r['eloff'][1:]
    gap, phase = r['gap_ms'][1:], r['phase'][1:]
    n = len(pix)

    print(f"\n{n} frames, eloff_every={int(r['eloff_every'])}, "
          f"extra gap 0..{float(r['gap_max_ms']):.0f} ms")
    print(f"START-to-START before block 0: median {np.nanmedian(gap):.1f} ms, "
          f"min {np.nanmin(gap):.1f}, max {np.nanmax(gap):.1f};  "
          f"between blocks 1..11: median {np.median(r['dt_blocks_ms']):.1f} ms")

    # Frame-to-frame flicker of each block relative to the mean of blocks
    # 1..11 of its half, so neither a common drift of the whole image nor
    # block 0 itself leaks into the reference:
    # rms of the change from one frame to the next, / sqrt(2) = per-frame noise
    print("\nFlicker per block [digits] = rms frame-to-frame change of "
          "(block mean - mean of blocks 1..11) / sqrt(2)")
    print("block | raw top  raw bot | pix-eloff top  pix-eloff bot")
    rel = {}
    for name, img in (('raw', pix), ('comp', comp)):
        m = block_means(img)
        rel[name] = m - m[:, :, 1:].mean(axis=2, keepdims=True)
    flick = {k: np.sqrt((np.diff(v, axis=0) ** 2).mean(axis=0) / 2)
             for k, v in rel.items()}
    for b in range(_BLOCKS):
        print(f"{b:5d} | {flick['raw'][0, b]:7.2f} {flick['raw'][1, b]:8.2f} | "
              f"{flick['comp'][0, b]:13.2f} {flick['comp'][1, b]:14.2f}")

    # Block 0 against the gap and the eloff phase
    b0 = rel['comp'][:, :, 0].mean(axis=1)           # top and bottom together
    g = gap
    if np.std(g) > 0:
        k, c0 = np.polyfit(g, b0, 1)
        print(f"\nBlock 0 (pix-eloff, rel. to blocks 1..11) vs. gap: "
              f"corr {np.corrcoef(g, b0)[0, 1]:+.2f}, slope {k:+.3f} digits/ms")
    print("Block 0 (pix-eloff, rel. to blocks 1..11) by frames since fresh eloff:")
    for p in np.unique(phase):
        sel = phase == p
        print(f"  phase {int(p)}: {b0[sel].mean():+8.2f} digits "
              f"(std {b0[sel].std():.2f}, n={sel.sum()})")

    print("\nPTAT / VDD per block: frame-to-frame rms change [digits], "
          "block 0 vs. blocks 1..11")
    for name in ('ptat', 'vdd'):
        a = r[name][1:].reshape(n, 2, _BLOCKS).astype(np.float64)
        s = np.sqrt((np.diff(a, axis=0) ** 2).mean(axis=(0, 1)) / 2)
        print(f"  {name}: block 0 {s[0]:.2f}, blocks 1..11 {s[1:].mean():.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog='SPI_FlickerTest.py')
    p.add_argument('--bus', type=int, default=0)
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--eloff_every', type=int, default=8)
    p.add_argument('--gap_ms', type=float, default=0.0)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--analyze', type=str, default=None)
    args = p.parse_args()

    if args.analyze:
        analyze(np.load(args.analyze))
    else:
        from hspy.drivers.spi import SPI_HTPA160x120dR1
        d = SPI_HTPA160x120dR1(bus=args.bus, device=0,
                               ee_enable=make_ee_enable(pin=8),
                               eloff_every=args.eloff_every)
        try:
            r = record(d, args.frames, args.gap_ms)
        finally:
            d.close()
        out = args.out or f"flicker_e{args.eloff_every}_g{args.gap_ms:g}.npz"
        np.savez_compressed(out, **r)
        print(f"Saved {out}")
        analyze(r)
