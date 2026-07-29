import numpy as np

def boxcar_shift_op(N, w, g):
    """H_g: length-N periodic signal -> N/w measurements, each = mean of window [j*w+g, j*w+g+w)."""
    m = N // w
    H = np.zeros((m, N))
    for j in range(m):
        for k in range(w):
            H[j, (j*w + g + k) % N] = 1.0 / w
    return H

def stacked_rank(N, w, shifts):
    H = np.vstack([boxcar_shift_op(N, w, g) for g in shifts])
    s = np.linalg.svd(H, compute_uv=False)
    smax = s.max()
    rank = int((s > 1e-9 * smax).sum())
    # effective rank via singular-value entropy
    p = s / s.sum(); eff = np.exp(-(p*np.log(p+1e-300)).sum())
    return rank, s, eff

for (N, w, label) in [(512, 8, "s=8 single-jump (64->512)"), (512, 2, "factor-2 octave")]:
    print(f"\n### N={N}, boxcar width w={w}   [{label}]  full dim={N}")
    all_shifts = list(range(w))
    for shifts in ([0], list(range(min(2,w))), all_shifts):
        rank, s, eff = stacked_rank(N, w, shifts)
        print(f"  shifts g in {shifts if len(shifts)<=4 else f'0..{shifts[-1]}'}: "
              f"rank={rank:4d}  nullity={N-rank:4d}  eff_rank={eff:7.1f}  "
              f"smin/smax={s[s>1e-9*s.max()].min()/s.max():.2e}")
