#!/usr/bin/env python
"""Thin out ``state_N.pt`` so a long run does not fill the quota.

map2map writes one checkpoint per epoch and NEVER removes the old ones, so the
footprint grows linearly: MEASURED 54.9 MB each, which is 1.5 GB at the 27
epochs reached so far and would be **56 GB** at the prescribed 1024.

**Dry run by default.** Deleting a checkpoint is irreversible and the only thing
standing between a 12 h training job and having to repeat it, so this prints what
it would remove and does nothing until ``--apply`` is passed.

The keep policy, and why each part exists:

* ``--keep-last N`` -- resume safety. ``checkpoint.pt`` is a symlink to the
  newest state, and the job resumes from it; keeping a few behind it means a
  corrupt or mid-write newest file is not fatal.
* ``--keep-every N`` -- the trend. The cutoff-vs-epoch diagnostic needs a ladder
  of checkpoints spanning training, not just the end. A regular stride is what
  makes ``k_half`` plottable against epoch afterwards.
* ``--keep`` -- named exceptions. ``state_1`` is the MSE-only model, the
  pre-adversarial reference every comparison is read against, and it is never
  regenerable from a later checkpoint.

Anything the policy does not name is removable, and the symlink target is
checked explicitly: a policy that deleted the file ``checkpoint.pt`` points at
would leave the next resubmission resuming from nothing and silently restarting
training from scratch.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

STATE = re.compile(r"^state_(\d+)\.pt$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a run directory")
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--keep-every", type=int, default=5)
    ap.add_argument("--keep", default="1",
                    help="comma-separated epochs to always keep (default: 1, "
                         "the MSE-only reference)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it this only reports")
    a = ap.parse_args(argv)

    d = Path(a.dir)
    states = {}
    for p in d.glob("state_*.pt"):
        m = STATE.match(p.name)
        if m:
            states[int(m.group(1))] = p
    if not states:
        print(f"no state_N.pt in {d}")
        return 0

    epochs = sorted(states)
    named = {int(x) for x in str(a.keep).replace(",", " ").split() if x.strip()}
    keep = set(named)
    keep |= set(epochs[-a.keep_last:]) if a.keep_last > 0 else set()
    if a.keep_every > 0:
        keep |= {e for e in epochs if e % a.keep_every == 0}

    # The resume target is sacred: deleting it makes the next job start over.
    link = d / "checkpoint.pt"
    if link.is_symlink() or link.exists():
        m = STATE.match(Path(link.resolve()).name)
        if m:
            keep.add(int(m.group(1)))

    drop = [e for e in epochs if e not in keep]
    mb = sum(states[e].stat().st_size for e in drop) / 1048576
    print(f"{d}")
    print(f"  {len(epochs)} checkpoints, epochs {epochs[0]}..{epochs[-1]}")
    print(f"  keep {len(keep)}: {sorted(keep)}")
    print(f"  drop {len(drop)}: {drop}")
    print(f"  reclaims {mb:,.0f} MB")
    if not a.apply:
        print("\nDRY RUN -- nothing deleted. Pass --apply to remove them.")
        return 0
    for e in drop:
        states[e].unlink()
    print(f"\ndeleted {len(drop)} checkpoints, {mb:,.0f} MB reclaimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
