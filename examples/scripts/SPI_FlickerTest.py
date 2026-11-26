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

Under the real load of spi-udp-threaded.py, at normal and at real-time
priority of the acquisition thread:
python ./SPI_FlickerTest.py --bus 0 --frames 300 --pipeline 192.168.178.22
python ./SPI_FlickerTest.py --bus 0 --frames 300 --pipeline 192.168.178.22 --rt_priority 20
python ./SPI_FlickerTest.py --bus 0 --frames 300 --pipeline 192.168.178.22 --rt_priority 20 --warmup

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
--pipeline : str (optional)
    IP address of a peer running udp-pickle-imshow-mp.py. Records inside the
    real threaded pipeline (acquisition, processing, UDP server) instead of
    a plain loop; --gap_ms is then ignored.
--target_port : int (default 5005)
    Port of that peer.
--rt_priority : int (optional)
    With --pipeline: SCHED_FIFO priority of the acquisition thread (see
    spi-udp-threaded.py for the one-time setup). Default: normal priority.
--warmup : flag
    Before block 0 of every frame, run one extra conversion of block 0 and
    discard it, so that block 0 follows a conversion like every other block
    instead of a longer idle time. Tests whether block 0 flickers because
    the sensor settles at the start of a frame.
--out : str (optional)
    File the recording is saved to (default flicker_e<n>_g<ms>.npz or
    flicker_e<n>_pipe_rt<priority>.npz, with _w appended for --warmup).
--analyze : str (optional)
    Skip recording, only analyse this .npz file.
"""

import argparse
import os
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


def _flicker(px: np.ndarray, ref: np.ndarray) -> tuple:
    """
    px (N, ...) pixel values per frame, ref (N,) reference per frame.
    Returns, each as rms frame-to-frame change / sqrt(2):
    the flicker of the mean of px minus ref, the part of it the independent
    noise of the pixels explains, and that noise of a single pixel.
    """
    d = np.diff(px.reshape(len(px), -1), axis=0)
    p = d.shape[1]
    c = d.mean(axis=1)
    e2 = ((d - c[:, None]) ** 2).mean()           # = sigma^2 (1 - 1/p) * 2
    c = c - np.diff(ref)
    return (np.sqrt((c ** 2).mean() / 2), np.sqrt(e2 / (p - 1) / 2),
            np.sqrt(e2 * p / (p - 1) / 2))


def make_ee_enable(pin: int = 8, chip: int = 0):
    import lgpio
    h = lgpio.gpiochip_open(chip)
    lgpio.gpio_claim_output(h, pin, 1)           # idle HIGH
    write = lgpio.gpio_write

    def ee_enable(level: bool) -> None:
        write(h, pin, level)
    ee_enable.handle = h
    return ee_enable


def record(d, n_frames: int, gap_ms: float = 0.0,
           target: tuple = None, rt_priority: int = None,
           warmup: bool = False) -> dict:
    """
    Record n_frames raw frames with driver d, see module docstring.

    target=None: a plain loop in this thread, with an extra random pause of
    0..gap_ms between frames. target=(ip, port): the real threaded pipeline
    of spi-udp-threaded.py (acquisition, processing, UDP server to target),
    with the acquisition thread at rt_priority; gap_ms is ignored.
    warmup: one discarded conversion of block 0 in front of every frame.
    """
    from hspy.drivers.spi.drivers import (_CMD_CONFIG, _BIT_START, _BIT_BLIND,
                                          _BIT_WAKEUP)

    if target is not None:
        import sys
        from queue import Queue
        from hspy.ipc.threads import (SPI_AcquisitionThread,
                                      SPI_ProcessingThread, UDP_PickleServer)
        sys.setswitchinterval(0.0005)            # as in spi-udp-threaded.py
        raw_buffer, spi_buffer = Queue(maxsize=4), Queue(maxsize=4)
        # Created before d._write_register is wrapped below: it works on a
        # pickled copy of d, and the wrapper cannot be pickled
        proc_thread = SPI_ProcessingThread(spi_driver=d,
                                           read_buffer=raw_buffer,
                                           write_buffer=spi_buffer)
        server_thread = UDP_PickleServer(name='udp_pickle_server_thread',
                                         read_buffer=spi_buffer,
                                         target_ip=target[0],
                                         target_port=target[1])

    # Time stamp every conversion START: (perf_counter, BLIND bit set)
    starts = []
    write_register = d._write_register

    def logged_write_register(cmd, value):
        if cmd == _CMD_CONFIG and value & _BIT_START:
            starts.append((time.perf_counter(), bool(value & _BIT_BLIND)))
        write_register(cmd, value)

    d._write_register = logged_write_register

    # One acquire_raw() plus its timing; runs in the acquiring thread
    raws, gap, dt_blocks, phase = [], [], [], []
    rt_actual = []          # real-time priority the acquiring thread got

    def grab() -> dict:
        if not rt_actual:
            fifo = (hasattr(os, 'sched_getscheduler')
                    and os.sched_getscheduler(0) == os.SCHED_FIFO)
            rt_actual.append(os.sched_getparam(0).sched_priority if fifo else -1)
        first = len(starts)
        if warmup:
            # Same steps as one block of read_frame(), data thrown away
            # (read_frame() overwrites the buffer again)
            d._write_register(_CMD_CONFIG, _BIT_WAKEUP | _BIT_START)
            time.sleep(0.9 * d._calib["t_block"] * 1e-3)
            d._wait_eoc()
            d._read_both_halves(d._io.xf_frame[0])
        raw = d.acquire_raw()
        phase.append(d._n_since_eloff)           # 1 = fresh eloff this frame

        # read_frame() runs before the BLIND conversion, so the first
        # START of this frame is block 0
        t_block = [t for t, blind in starts[first:] if not blind]
        if warmup:
            # gap = from the warm-up START to the START of block 0
            gap.append(t_block[1] - t_block[0])
            t_block = t_block[1:]
        else:
            gap.append(t_block[0] - starts[first - 1][0] if first else np.nan)
        dt_blocks.append(np.median(np.diff(t_block)))
        raws.append((raw['pix_raw'], raw['eloff_raw']))   # references only
        return raw

    try:
        if target is None:
            for i in range(n_frames):
                grab()
                if gap_ms > 0:
                    time.sleep(random.uniform(0, gap_ms) * 1e-3)
        else:
            class RecordingAcquisitionThread(SPI_AcquisitionThread):
                def _acquire(self):
                    return grab()

            spi_thread = RecordingAcquisitionThread(spi_driver=d,
                                                    write_buffer=raw_buffer,
                                                    rt_priority=rt_priority)
            threads = (spi_thread, proc_thread, server_thread)
            for t in threads:
                t.start()
            try:
                while len(raws) < n_frames and spi_thread.is_alive():
                    time.sleep(0.2)
            finally:
                for t in reversed(threads):
                    t.stop()
                for t in threads:
                    t.join()
    finally:
        d._write_register = write_register

    # Convert after the recording, so that it adds no load while recording
    n = min(n_frames, len(raws))
    pix, eloff, ptat, vdd = [], [], [], []
    for pix_raw, eloff_raw in raws[:n]:
        data = d.convert_spi_data({'pix_raw': pix_raw, 'eloff_raw': eloff_raw})
        pix.append(data['pix'].copy())
        eloff.append(data['eloff'].copy())
        ptat.append(data['ptat'].copy())
        vdd.append(data['vdd'].copy())

    return dict(pix=np.array(pix), eloff=np.array(eloff),
                ptat=np.array(ptat), vdd=np.array(vdd),
                gap_ms=np.array(gap[:n]) * 1e3,
                dt_blocks_ms=np.array(dt_blocks[:n]) * 1e3,
                phase=np.array(phase[:n]), eloff_every=d.eloff_every,
                gap_max_ms=gap_ms if target is None else np.nan,
                pipeline=target is not None,
                rt_priority=rt_actual[0] if rt_actual else -1,
                warmup=warmup)


def analyze(r) -> None:
    # Frame 0 is the first one after init() and has no gap: left out
    pix = r['pix'][1:].astype(np.float64)
    comp = pix - r['eloff'][1:]
    gap, phase = r['gap_ms'][1:], r['phase'][1:]
    n = len(pix)

    if 'pipeline' in getattr(r, 'files', r) and r['pipeline']:
        rt = int(r['rt_priority'])
        mode = (f"threaded pipeline, acquisition priority "
                f"{'normal' if rt < 0 else f'SCHED_FIFO {rt}'}")
    else:
        mode = f"extra gap 0..{float(r['gap_max_ms']):.0f} ms"
    if 'warmup' in getattr(r, 'files', r) and r['warmup']:
        mode += ", warm-up conversion before block 0"
    print(f"\n{n} frames, eloff_every={int(r['eloff_every'])}, {mode}")
    print(f"START-to-START before block 0: median {np.nanmedian(gap):.1f} ms, "
          f"min {np.nanmin(gap):.1f}, max {np.nanmax(gap):.1f}, "
          f"std {np.nanstd(gap):.2f};  "
          f"between blocks 1..11: median {np.median(r['dt_blocks_ms']):.1f} ms")

    # Frame-to-frame flicker of each block relative to the mean of blocks
    # 1..11 of its half, so neither a common drift of the whole image nor
    # block 0 itself leaks into the reference. Electrical offsets are the
    # same for every block and cancel here, so only pix-eloff is shown.
    # The flicker of a block mean has two parts: a common mode all pixels
    # of the block share (seen as flicker of the block) and the independent
    # noise of its pixels averaged over the block, larger the noisier the
    # pixels are (corners). The latter is estimated from the scatter of the
    # pixels around their block mean and taken out.
    img = comp.reshape(n, _ROWS, _COLS)
    ref = np.stack([img[:, (_ROW_HALF == h) & (_ROW_BLOCK >= 1)].mean(axis=(1, 2))
                    for h in range(2)], axis=1)                     # (n, 2)
    tot, flo, sig = (np.empty((2, _BLOCKS)) for _ in range(3))
    for h in range(2):
        for b in range(_BLOCKS):
            tot[h, b], flo[h, b], sig[h, b] = _flicker(
                img[:, (_ROW_HALF == h) & (_ROW_BLOCK == b)], ref[:, h])

    # Tilt: a line across the 12 blocks of a half whose slope changes from
    # frame to frame (an offset drifting during the frame, as the blocks are
    # converted one after the other). It is fitted per frame to blocks 1..11
    # only, so that block 0 does not pull it, and removed from all blocks.
    m = block_means(comp)
    rel = m - m[:, :, 1:].mean(axis=2, keepdims=True)
    x = np.arange(_BLOCKS) - (_BLOCKS - 1) / 2
    xc = x - x[1:].mean()
    sxx = (xc[1:] ** 2).sum()
    slope = (rel[:, :, 1:] * xc[1:]).sum(axis=2) / sxx              # (n, 2)
    res = rel - rel[:, :, 1:].mean(axis=2, keepdims=True) - slope[:, :, None] * xc
    tot_res = np.sqrt((np.diff(res, axis=0) ** 2).mean(axis=0) / 2)
    # Pixel noise left in the residual: less for the fitted blocks 1..11,
    # more for the extrapolated block 0
    lev = 1 / (_BLOCKS - 1) + xc ** 2 / sxx
    lev[0] = -lev[0]

    def excess(t, f):
        return np.sqrt(np.maximum(t ** 2 - f ** 2, 0))

    print("\nFlicker per block [digits], pix-eloff rel. to blocks 1..11, "
          "rms frame-to-frame change / sqrt(2):")
    print("  pixel  = noise of a single pixel (mean over the block)")
    print("  total  = flicker of the block mean")
    print("  noise  = part of it explained by independent pixel noise")
    print("  common = sqrt(total^2 - noise^2), flicker shared by the block")
    print("  detilt = common, after removing the per-frame tilt across blocks")
    print("      |            top half               |            bottom half")
    print("block | pixel total noise common detilt | pixel total noise common detilt")
    for b in range(_BLOCKS):
        print(f"{b:5d} |" + " |".join(
            f"{sig[h, b]:6.1f}{tot[h, b]:6.2f}{flo[h, b]:6.2f}"
            f"{excess(tot[h, b], flo[h, b]):7.2f}"
            f"{excess(tot_res[h, b], flo[h, b] * np.sqrt(1 - lev[b])):7.2f}"
            for h in range(2)))
    ds = np.diff(slope.mean(axis=1))
    print(f"Tilt: rms frame-to-frame change of block 11 - block 0 = "
          f"{np.sqrt((ds ** 2).mean() / 2) * (_BLOCKS - 1):.2f} digits; "
          f"corr. top/bottom {np.corrcoef(slope[:, 0], slope[:, 1])[0, 1]:+.2f}")
    s = slope.mean(axis=1)
    cs = [f"{name} {np.corrcoef(s, v)[0, 1]:+.2f}" for name, v in
          (('gap', gap),
           ('PTAT', r['ptat'][1:].reshape(n, -1).mean(axis=1)),
           ('VDD', r['vdd'][1:].reshape(n, -1).mean(axis=1)))
          if np.std(v) > 0]
    print(f"Tilt slope corr. with {', '.join(cs)}")

    # Block 0 row by row, in the order of the rows in the image
    print("\nBlock 0 row by row [digits] (common = flicker shared by the row):")
    print("image row | pixel total noise common")
    for row in list(range(_ROWS_BLOCK)) + list(range(_ROWS - _ROWS_BLOCK, _ROWS)):
        t, f, sp = _flicker(img[:, row:row + 1], ref[:, _ROW_HALF[row]])
        print(f"{row:9d} |{sp:6.1f}{t:6.2f}{f:6.2f}{excess(t, f):7.2f}")

    # Block 0 against the gap and the eloff phase
    b0 = rel[:, :, 0].mean(axis=1)           # top and bottom together
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
    p.add_argument('--pipeline', type=str, default=None, metavar='TARGET_IP')
    p.add_argument('--target_port', type=int, default=5005)
    p.add_argument('--rt_priority', type=int, default=None)
    p.add_argument('--warmup', action='store_true')
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
        target = (args.pipeline, args.target_port) if args.pipeline else None
        try:
            r = record(d, args.frames, args.gap_ms, target, args.rt_priority,
                       args.warmup)
        finally:
            d.close()
        if target:
            name = f"pipe_rt{args.rt_priority or 0}"
        else:
            name = f"g{args.gap_ms:g}"
        if args.warmup:
            name += "_w"
        out = args.out or f"flicker_e{args.eloff_every}_{name}.npz"
        np.savez_compressed(out, **r)
        print(f"Saved {out}")
        analyze(r)
