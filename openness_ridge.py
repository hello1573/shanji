# -*- coding: utf-8 -*-
"""
Openness-only ridge extraction, v2 (pure numpy). No inverse-terrain, no flow.

Why v2: the old top-hat (P = Rs - bg) at a single scale SUPPRESSES the most
obvious ridges -- a broad/long dominant ridge fills its own background window,
so Rs ~= bg and P ~= 0 right on the crest. And the old greedy single-track walk
breaks at saddles and cannot represent branching spur networks.

v2 fixes both:
  1. SIGNAL: multi-scale ridge strength. Compute the top-hat at several
     background scales and take the max, so both sharp and broad ridges score high.
  2. TOPOLOGY: hysteresis threshold (high seed + low grow) -> one connected ridge
     mask that does not break at saddles -> Zhang-Suen thinning to 1px centerlines
     -> vectorize into a branching polyline network -> prune hair spurs -> rank
     whole ridge systems by length * mean-strength and keep the top N.

All thresholds adapt per tile to the positive-strength distribution (percentiles)
with absolute floors, so tiles of different openness magnitude behave the same.

Main entry: extract_openness_ridges(...). transform=None -> (row,col) lines;
rasterio Affine -> (x,y) lines.
"""
import numpy as np


# ----------------------------------------------------------------------------
# NaN-aware smoothing
# ----------------------------------------------------------------------------
def _shift_sum(fill, w, k):
    rows, cols = fill.shape
    sfill = np.zeros_like(fill)
    sw = np.zeros_like(w)
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            r0 = max(0, dr); r1 = rows + min(0, dr)
            c0 = max(0, dc); c1 = cols + min(0, dc)
            sr0 = max(0, -dr); sr1 = rows + min(0, -dr)
            sc0 = max(0, -dc); sc1 = cols + min(0, -dc)
            sfill[sr0:sr1, sc0:sc1] += fill[r0:r1, c0:c1]
            sw[sr0:sr1, sc0:sc1] += w[r0:r1, c0:c1]
    return sfill, sw


def smooth_nan(a, valid, iters=2, k=1):
    """NaN-aware box smoothing over the valid mask."""
    out = np.where(valid, a.astype(float), 0.0)
    w0 = valid.astype(float)
    for _ in range(max(1, iters)):
        sfill, sw = _shift_sum(out, w0, k)
        out = np.where(sw > 0, sfill / np.maximum(sw, 1e-9), 0.0)
    return np.where(valid, out, np.nan)


# ----------------------------------------------------------------------------
# Multi-scale ridge strength (the fix for dominant broad ridges)
# ----------------------------------------------------------------------------
def ridge_strength(openness, support, smooth_iters=2, smooth_k=1,
                   bg_scales=((6, 2), (15, 3), (40, 4))):
    """Return P: multi-scale ridge prominence (max of top-hats at several scales).

    Rs        = smoothed openness (bright = ridge)
    bg_s      = broad background at scale s
    P         = max_s (Rs - bg_s)   # strong on both sharp and broad ridges
    """
    Rs = smooth_nan(openness, support, iters=smooth_iters, k=smooth_k)
    P = None
    for (b_iters, b_k) in bg_scales:
        bg = smooth_nan(Rs, support, iters=b_iters, k=b_k)
        ph = Rs - bg
        P = ph if P is None else np.maximum(P, ph)
    P[~support] = np.nan
    return P, Rs


def _pos_percentile(P, support, pct, floor):
    pos = P[support & np.isfinite(P) & (P > 0)]
    if pos.size == 0:
        return float(floor)
    return max(float(floor), float(np.percentile(pos, pct)))


# ----------------------------------------------------------------------------
# Connected-component labeling (union-find, 8-connectivity)
# ----------------------------------------------------------------------------
def _label(mask, conn=8):
    H, W = mask.shape
    ys, xs = np.where(mask)
    n = ys.size
    out = -np.ones((H, W), dtype=np.int64)
    if n == 0:
        return out, 0
    idx = -np.ones((H, W), dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    parent = np.arange(n)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    offs = ((0, 1), (1, 0)) if conn == 4 else ((0, 1), (1, 0), (1, 1), (1, -1))
    for i in range(n):
        y = int(ys[i]); x = int(xs[i]); a = int(idx[y, x])
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                union(a, int(idx[ny, nx]))
    roots = np.array([find(i) for i in range(n)])
    uniq = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
    for i in range(n):
        out[ys[i], xs[i]] = uniq[roots[i]]
    return out, len(uniq)


# ----------------------------------------------------------------------------
# Fill small enclosed holes (removes ring-shaped small ridges of any topology)
# ----------------------------------------------------------------------------
def _fill_small_holes(mask, max_hole_area):
    """Fill enclosed background holes whose area <= max_hole_area.

    A ridge loop/ring encloses a small background hole. Filling that hole turns
    the ring into a solid blob, which thins to a short stub and is then dropped
    by the length/span filters -- so closed-ring artifacts disappear entirely,
    regardless of how many segments form the ring.

    Background is labelled with 4-connectivity so an 8-connected ridge loop
    properly seals its interior (digital Jordan-curve duality); with
    8-connectivity the background would leak out through diagonal gaps.
    """
    if max_hole_area <= 0:
        return mask
    bg = ~mask
    lab, n = _label(bg, conn=4)
    if n == 0:
        return mask
    H, W = mask.shape
    outside = set()
    border = np.concatenate([lab[0, :], lab[H - 1, :], lab[:, 0], lab[:, W - 1]])
    for v in np.unique(border):
        if v >= 0:
            outside.add(int(v))
    counts = np.bincount(lab[lab >= 0].ravel(), minlength=n)
    out = mask.copy()
    for cid in range(n):
        if cid in outside:
            continue
        if int(counts[cid]) <= max_hole_area:
            out[lab == cid] = True
    return out


# ----------------------------------------------------------------------------
# Hysteresis: keep low-threshold components that contain a high-threshold seed
# ----------------------------------------------------------------------------
def hysteresis_mask(P, support, seed_th, grow_th):
    low = support & np.isfinite(P) & (P >= grow_th)
    lab, k = _label(low)
    if k == 0:
        return low
    seed = low & (P >= seed_th)
    good = set(int(v) for v in np.unique(lab[seed]) if v >= 0)
    if not good:
        return np.zeros_like(low)
    keep = np.isin(lab, list(good)) & low
    return keep


# ----------------------------------------------------------------------------
# Zhang-Suen thinning to 1px skeleton (vectorized)
# ----------------------------------------------------------------------------
def _neighbors8(img):
    P = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    P[1:-1, 1:-1] = img
    p2 = P[0:-2, 1:-1]   # N
    p3 = P[0:-2, 2:]     # NE
    p4 = P[1:-1, 2:]     # E
    p5 = P[2:, 2:]       # SE
    p6 = P[2:, 1:-1]     # S
    p7 = P[2:, 0:-2]     # SW
    p8 = P[1:-1, 0:-2]   # W
    p9 = P[0:-2, 0:-2]   # NW
    return p2, p3, p4, p5, p6, p7, p8, p9


def zhang_suen(mask):
    img = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors8(img)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = np.zeros_like(B)
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                c1 = (p2 * p4 * p6 == 0)
                c2 = (p4 * p6 * p8 == 0)
            else:
                c1 = (p2 * p4 * p8 == 0)
                c2 = (p2 * p6 * p8 == 0)
            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if cond.any():
                img[cond] = 0
                changed = True
    return img.astype(bool)


# ----------------------------------------------------------------------------
# Skeleton -> polyline segments (split at endpoints/junctions)
# ----------------------------------------------------------------------------
def _skel_graph(skel):
    pts = [(int(r), int(c)) for r, c in zip(*np.where(skel))]
    ptset = set(pts)

    def nbrs(p):
        y, x = p
        r = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    q = (y + dy, x + dx)
                    if q in ptset:
                        r.append(q)
        return r

    deg = {p: len(nbrs(p)) for p in pts}
    return ptset, nbrs, deg


def skeleton_segments(skel):
    ptset, nbrs, deg = _skel_graph(skel)
    nodes = set(p for p in ptset if deg[p] != 2)
    used = set()

    def ekey(a, b):
        return (a, b) if a <= b else (b, a)

    segments = []
    # 1) segments anchored at nodes (endpoints / junctions)
    for s in sorted(nodes):
        for nb in sorted(nbrs(s)):
            if ekey(s, nb) in used:
                continue
            seg = [s]; prev = s; cur = nb
            used.add(ekey(prev, cur)); seg.append(cur)
            while cur not in nodes:
                cand = [q for q in nbrs(cur) if q != prev and ekey(cur, q) not in used]
                if not cand:
                    break
                nx = sorted(cand)[0]
                used.add(ekey(cur, nx)); prev, cur = cur, nx; seg.append(cur)
            segments.append(seg)
    # 2) pure loops (all degree 2, no nodes)
    for s in sorted(ptset):
        for nb in sorted(nbrs(s)):
            if ekey(s, nb) in used:
                continue
            seg = [s]; prev = s; cur = nb
            used.add(ekey(prev, cur)); seg.append(cur)
            while cur != s:
                cand = [q for q in nbrs(cur) if q != prev and ekey(cur, q) not in used]
                if not cand:
                    break
                nx = sorted(cand)[0]
                used.add(ekey(cur, nx)); prev, cur = cur, nx; seg.append(cur)
            if len(seg) >= 2:
                segments.append(seg)
    return segments, deg


def _polylen(seg):
    a = np.asarray(seg, dtype=float)
    if len(a) < 2:
        return 0.0
    return float(np.sqrt((np.diff(a, axis=0) ** 2).sum(axis=1)).sum())


def _prune_spurs(segments, deg, min_spur_len):
    """Drop short hair spurs: a segment with a degree-1 endpoint and length below
    min_spur_len. Interior connectors (both endpoints are junctions) are kept."""
    if min_spur_len <= 0:
        return segments
    keep = []
    for seg in segments:
        a, b = seg[0], seg[-1]
        has_tip = (deg.get(a, 0) == 1) or (deg.get(b, 0) == 1)
        if has_tip and _polylen(seg) < min_spur_len:
            continue
        keep.append(seg)
    return keep


def _rebuild_skel(segments, shape):
    new = np.zeros(shape, dtype=bool)
    for seg in segments:
        for (r, c) in seg:
            new[r, c] = True
    return new


def _prune_spurs_iter(skel, min_spur_len, max_iter=25):
    """Iteratively remove short hair spurs. After a spur is removed its junction
    may drop to degree 2 and expose a new short stub, so repeat until stable."""
    if min_spur_len <= 0:
        return skel
    cur = skel
    for _ in range(max_iter):
        segments, deg = skeleton_segments(cur)
        kept = _prune_spurs(segments, deg, min_spur_len)
        if len(kept) == len(segments):
            break
        new = _rebuild_skel(kept, cur.shape)
        if int(new.sum()) == int(cur.sum()):
            break
        cur = new
    return cur


def _break_small_loops(segments, deg, max_loop_len):
    """Remove ring artifacts. Drop small pure loops (closed segments) and open
    small 2-edge cycles (two segments joining the same node pair) by dropping the
    shorter edge. Large loops are left intact."""
    if max_loop_len <= 0:
        return segments
    out = []
    by_pair = {}
    for seg in segments:
        a, b = seg[0], seg[-1]
        if a == b:
            # pure closed loop: drop if small
            if _polylen(seg) >= max_loop_len:
                out.append(seg)
            continue
        key = (a, b) if a <= b else (b, a)
        by_pair.setdefault(key, []).append(seg)
    for _key, segs in by_pair.items():
        if len(segs) >= 2:
            segs_sorted = sorted(segs, key=_polylen)
            out.append(segs_sorted[-1])           # keep the longest path
            for s in segs_sorted[:-1]:            # the rest are parallel edges
                if _polylen(s) >= max_loop_len:  # keep only if the loop is large
                    out.append(s)
        else:
            out.append(segs[0])
    return out


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def extract_openness_ridges(openness, support, transform=None,
                            smooth_iters=2, smooth_k=1,
                            bg_scales=((6, 2), (15, 3), (40, 4)),
                            seed_pct=90.0, prom_seed=0.0,
                            prom_continue_pct=65.0, prom_continue=0.0,
                            min_mean_prom_pct=80.0, min_mean_prom=0.0,
                            min_length_cells=120, keep_top_n=0,
                            prune_spur_cells=8, min_span_cells=0,
                            max_loop_cells=0, max_hole_cells=0,
                            edge_len_frac=1.0, edge_margin=3,
                            min_ridge_spacing=0, max_parallel_overlap=0.5):
    """Extract ridge centerlines from an openness raster (openness only).

    Returns a list of polylines. Each polyline is an (N,2) array:
      - transform=None        -> (row, col) pixel coordinates (for tests)
      - rasterio Affine given  -> (x, y) projected coordinates
    Branching ridge systems are returned as several polylines that share junctions.

    Adaptive thresholds (per tile, percentiles of positive strength, with floors):
      - seed_pct / prom_seed         : high (seed) threshold of the hysteresis
      - prom_continue_pct / prom_continue : low (grow) threshold; bridges saddles
      - min_mean_prom_pct / min_mean_prom : a whole ridge system is dropped if its
                                            mean strength is below this
    Geometry / selection:
      - min_length_cells : drop ridge systems whose total length is below this
      - min_span_cells   : drop ridge systems whose spatial extent (bbox diagonal)
                           is below this; removes compact small-ridge noise clusters
                           that a pure length filter cannot catch
      - prune_spur_cells : drop hair spurs shorter than this from the skeleton
                           (applied ITERATIVELY: removes multi-level stubs)
      - max_loop_cells   : remove ring artifacts whose loop length is below this
                           (drops small closed loops, opens small 2-edge cycles)
      - max_hole_cells   : fill enclosed holes (area in cells) at/below this size
                           BEFORE thinning; collapses ring-shaped small ridges of
                           ANY topology to a short stub that the length/span
                           filters then drop (the robust fix for circle artifacts)
      - keep_top_n       : keep only the N strongest ridge systems (0 = keep all)
      - edge_len_frac    : leniency for ridge systems that TOUCH the data boundary
                           (array edge or no-data boundary). Such systems are
                           legitimately truncated, so their length / span /
                           mean-strength thresholds are multiplied by this factor.
                           This recovers real ridges cut off at the tile edge
                           WITHOUT loosening thresholds for interior ridges.
                           1.0 = no leniency (default); 0.5 = half thresholds at edge.
      - edge_margin      : how many cells inward from the data boundary still count
                           as "edge". A truncated ridge's skeleton usually ends a
                           few cells short of the exact boundary (thinning erosion),
                           so a thin band is used instead of the 1-px boundary line.
      - min_ridge_spacing: spatial non-max suppression. In dense fans of near-
                           parallel ridges, keep only the dominant (strongest)
                           ridge and drop weaker ones running within this many
                           cells of it. 0 = off. Crossing / branching ridges that
                           only touch at junctions are NOT suppressed.
      - max_parallel_overlap: a candidate ridge is suppressed only if more than
                           this fraction of its length falls within the spacing
                           band of an already-kept stronger ridge (so it runs
                           alongside it). Lower = suppress more aggressively.
    """
    support = support.astype(bool)
    support = support & np.isfinite(openness)   # guard against NaN openness

    P, Rs = ridge_strength(openness, support, smooth_iters, smooth_k, bg_scales)

    seed_th = _pos_percentile(P, support, seed_pct, prom_seed)
    grow_th = _pos_percentile(P, support, prom_continue_pct, prom_continue)
    mean_th = _pos_percentile(P, support, min_mean_prom_pct, min_mean_prom)

    mask = hysteresis_mask(P, support, seed_th, grow_th)
    if not mask.any():
        return []
    mask = _fill_small_holes(mask, max_hole_cells)
    skel = zhang_suen(mask)
    skel = _prune_spurs_iter(skel, prune_spur_cells)
    segments, deg = skeleton_segments(skel)
    segments = _break_small_loops(segments, deg, max_loop_cells)
    segments = [s for s in segments if len(s) >= 2]
    if not segments:
        return []

    # group surviving segments into ridge systems (connected components)
    keep_mask = np.zeros(P.shape, dtype=bool)
    for seg in segments:
        for (r, c) in seg:
            keep_mask[r, c] = True
    lab, ncomp = _label(keep_mask)

    comp_segs = {}
    for seg in segments:
        r0, c0 = seg[0]
        cid = int(lab[r0, c0])
        comp_segs.setdefault(cid, []).append(seg)

    # cells in the valid region that border the data boundary (the array frame or
    # the support/no-data boundary). A system touching this is likely truncated.
    pad = np.pad(support, 1, mode="constant", constant_values=False)
    outside_nb = ((~pad[0:-2, 1:-1]) | (~pad[2:, 1:-1]) |
                  (~pad[1:-1, 0:-2]) | (~pad[1:-1, 2:]))
    border = support & outside_nb
    # grow the 1-px boundary into a thin band inward (a truncated ridge's skeleton
    # typically stops a few cells short of the exact data edge after thinning)
    band = border.copy()
    for _ in range(int(max(edge_margin, 0))):
        gnb = np.zeros_like(band)
        gnb[1:, :] |= band[:-1, :]
        gnb[:-1, :] |= band[1:, :]
        gnb[:, 1:] |= band[:, :-1]
        gnb[:, :-1] |= band[:, 1:]
        band |= gnb & support

    systems = []
    for cid, segs in comp_segs.items():
        cells = np.argwhere(lab == cid)
        pv = P[cells[:, 0], cells[:, 1]]
        mean_prom = float(np.nanmean(pv)) if pv.size else 0.0
        total_len = float(sum(_polylen(s) for s in segs))
        if cells.size:
            rmn, cmn = cells.min(axis=0)
            rmx, cmx = cells.max(axis=0)
            span = float(np.hypot(rmx - rmn, cmx - cmn))
        else:
            span = 0.0
        # truncated edge systems get more lenient thresholds
        on_edge = bool(band[cells[:, 0], cells[:, 1]].any())
        f = edge_len_frac if on_edge else 1.0
        if total_len < min_length_cells * f:
            continue
        if span < min_span_cells * f:
            continue
        if mean_prom < mean_th * f:
            continue
        strength = total_len * max(mean_prom, 0.0)
        systems.append((strength, total_len, cid, segs))

    # deterministic ordering: by strength desc, then component id
    systems.sort(key=lambda t: (-t[0], t[2]))

    # spatial non-max suppression: in dense fans of near-parallel ridges, accept
    # the strongest first and drop weaker neighbors that run alongside it within
    # min_ridge_spacing cells. Crossing/branching ridges only overlap at a few
    # junction cells, so their overlap fraction stays low and they are kept.
    if min_ridge_spacing and min_ridge_spacing > 0 and systems:
        occupied = np.zeros(P.shape, dtype=bool)
        kept = []
        for item in systems:
            segs = item[3]
            rr = np.concatenate([np.asarray(s)[:, 0] for s in segs]).astype(int)
            cc = np.concatenate([np.asarray(s)[:, 1] for s in segs]).astype(int)
            if rr.size and float(occupied[rr, cc].mean()) > max_parallel_overlap:
                continue
            kept.append(item)
            m = np.zeros(P.shape, dtype=bool)
            m[rr, cc] = True
            for _ in range(int(min_ridge_spacing)):
                g = np.zeros_like(m)
                g[1:, :] |= m[:-1, :]
                g[:-1, :] |= m[1:, :]
                g[:, 1:] |= m[:, :-1]
                g[:, :-1] |= m[:, 1:]
                m |= g
            occupied |= m
        systems = kept

    if keep_top_n and keep_top_n > 0:
        systems = systems[:keep_top_n]

    out = []
    for _st, _tl, _cid, segs in systems:
        for seg in sorted(segs, key=lambda s: (-_polylen(s), s[0])):
            arr = np.asarray(seg, dtype=float)
            if transform is None:
                out.append(arr)
            else:
                xs = transform.c + (arr[:, 1] + 0.5) * transform.a
                ys = transform.f + (arr[:, 0] + 0.5) * transform.e
                out.append(np.column_stack([xs, ys]))
    return out
