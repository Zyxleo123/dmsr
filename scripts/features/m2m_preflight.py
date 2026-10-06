#!/usr/bin/env python
"""Fail a map2map run in seconds, on the login node, instead of in the queue.

Two submissions of the baseline died before a single training step, each on a
one-line problem that a GPU allocation was spent discovering:

* job 65050 -- ``--optimizer-args: invalid loads value: '{"betas": [0, 0.99]'``.
  A bash ``${VAR:=default}`` ends at the first unescaped ``}``, so the JSON
  default was truncated. Died at argparse, ~12 seconds in, after a 4-hour CPU
  split had already run.
* job 66044 -- ``betas must be either both floats or both Tensors``. ``[0, 0.99]``
  parses as ``(int, float)`` and torch 2.11's Adam rejects the mix. Died at
  optimizer construction.

A third was waiting: ``train.py`` calls ``np.float(style)``, removed in numpy
2.0, and that line only runs once the adversary starts at ``--adv-start``. On
this data that is the second epoch, so it would have surfaced hours in.

This runs the same four stages a real job passes through, on CPU, with no
allocation:

1. **argparse** -- their own parser on the real argument list
2. **version aliases** -- grep their source for numpy-2 removals
3. **dataset** -- build their ``FieldDataset`` and pull one sample, which is
   where a wrong layout, a missing style file or a mismatched box set shows up
4. **model + optimizer** -- construct both, which is where the betas failed

It imports map2map rather than reimplementing any of it, so a check passing here
means their code accepted the configuration, not that ours thinks it should.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import traceback
from pathlib import Path
from typing import List

NUMPY2_REMOVED = re.compile(r"\bnp\.(float|int|bool|object|str)\b(?!\w)")


def stage(name: str, n: int, total: int) -> None:
    print(f"\n[{n}/{total}] {name}", flush=True)


def check_aliases(repo: Path) -> List[str]:
    """numpy-2 removed `np.float` and friends; these raise only when executed."""
    bad = []
    for f in sorted(repo.rglob("*.py")):
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if NUMPY2_REMOVED.search(line):
                bad.append(f"{f.relative_to(repo)}:{i}: {line.strip()}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="the map2map checkout")
    ap.add_argument("--args-file",
                    help="file holding the m2m.py train arguments, one per line")
    ap.add_argument("--args-string", help="the same, as one shell-quoted string")
    ap.add_argument("--skip-model", action="store_true",
                    help="skip stage 4 (it allocates the model on CPU)")
    a = ap.parse_args(argv)

    repo = Path(a.repo).resolve()
    if not (repo / "m2m.py").is_file():
        print(f"FAIL: no m2m.py at {repo}")
        return 1
    sys.path.insert(0, str(repo))

    if a.args_file:
        raw = Path(a.args_file).read_text()
        argv_train = [l for l in (x.strip() for x in raw.splitlines())
                      if l and not l.startswith("#")]
    elif a.args_string:
        argv_train = shlex.split(a.args_string)
    else:
        print("FAIL: need --args-file or --args-string")
        return 1

    total, failed = 4, []

    # -- 1. their argparse ---------------------------------------------------
    stage("argparse (their parser, the real argument list)", 1, total)
    try:
        from map2map.args import get_args
        sys.argv = ["m2m.py", "train"] + argv_train
        args = get_args()
        print(f"    ok: model={args.model} adv={args.adv_model} "
              f"crop={args.crop} in_pad={args.in_pad} "
              f"scale={args.scale_factor} batch={args.batch_size}")
        print(f"    optimizer_args={args.optimizer_args!r}")
    except SystemExit as e:
        print(f"    FAIL: argparse rejected the arguments (exit {e.code})")
        return 1
    except Exception:
        traceback.print_exc()
        return 1

    # -- 2. version aliases --------------------------------------------------
    stage("numpy-2 removed aliases in their source", 2, total)
    bad = check_aliases(repo / "map2map")
    if bad:
        failed.append("numpy aliases")
        print("    FAIL: these raise AttributeError under numpy >= 2, and some "
              "only run\n          once the adversary starts:")
        for b in bad:
            print(f"      {b}")
    else:
        print("    ok: none")

    # -- 3. their dataset ----------------------------------------------------
    stage("FieldDataset: build and pull one sample", 3, total)
    try:
        from map2map.data import FieldDataset
        ds = FieldDataset(
            in_patterns=args.train_in_patterns,
            tgt_patterns=args.train_tgt_patterns,
            style_pattern=args.train_style_pattern,
            in_norms=args.in_norms, tgt_norms=args.tgt_norms,
            callback_at=args.callback_at,
            augment=args.augment, aug_shift=args.aug_shift,
            crop=args.crop, crop_start=args.crop_start,
            crop_stop=args.crop_stop, crop_step=args.crop_step,
            in_pad=args.in_pad, tgt_pad=args.tgt_pad,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )
        d = ds[0]
        print(f"    ok: {len(ds)} samples, in_chan={ds.in_chan} "
              f"tgt_chan={ds.tgt_chan} style_size={ds.style_size}")
        print(f"        input {tuple(d['input'].shape)}  "
              f"target {tuple(d['target'].shape)}  style {d['style'].tolist()}")
        if any(c != 3 for c in ds.in_chan):
            print("    WARNING: a field with != 3 channels. map2map negates "
                  "vector components\n             only when channels == ndim, "
                  "so --augment would flip the voxels and\n             leave "
                  "the components pointing the old way.")
    except Exception:
        failed.append("dataset")
        traceback.print_exc()

    # -- 4. model + optimizer ------------------------------------------------
    stage("model and optimizer construction", 4, total)
    if a.skip_model:
        print("    skipped (--skip-model)")
    else:
        try:
            import torch
            from map2map import models
            from map2map.utils import import_attr

            m = import_attr(args.model, models, callback_at=args.callback_at)
            m = m(sum(ds.in_chan), sum(ds.tgt_chan), style_size=ds.style_size,
                  scale_factor=args.scale_factor, **args.misc_kwargs)
            opt = import_attr(args.optimizer, torch.optim,
                              callback_at=args.callback_at)
            opt = opt(m.parameters(), lr=args.lr, **args.optimizer_args)
            print(f"    ok: {args.model} "
                  f"{sum(p.numel() for p in m.parameters())} params; "
                  f"{args.optimizer}(lr={args.lr}, {args.optimizer_args})")
            # The scheduler. map2map's --scheduler-args DEFAULTS to
            # '{"verbose": true}', and torch removed ReduceLROnPlateau's
            # `verbose`. That killed job 67104 at setup, AFTER the optimizer --
            # which is exactly as far as this stage used to reach.
            torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **args.scheduler_args)
            print(f"    ok: ReduceLROnPlateau({args.scheduler_args})")

            if args.adv_model:
                d_ = import_attr(args.adv_model, models,
                                 callback_at=args.callback_at)
                # DO NOT reimplement the channel arithmetic. It was wrong
                # twice: lag2eul returns EIGHT channels here (eul_scale_factor
                # defaults to 2, so a 64^3 field paints to 128^3 and comes back
                # inverse-pixel-shuffled), and styled_srsgan.D adds those 8
                # itself in block0 with its own hard-coded `in_chan + 8`.
                # Guessing produced a "+1 missing" verdict that was false and
                # that I then patched into their code, breaking a working run.
                #
                # So: build D exactly as train.py does, build the tensor
                # exactly as the training loop does, and push it through. A
                # forward pass is the only check that cannot be wrong about
                # this.
                built = (sum(ds.in_chan + ds.tgt_chan) if args.cgan
                         else sum(ds.tgt_chan))
                d_ = d_(built, 1, style_size=ds.style_size,
                        scale_factor=args.scale_factor, **args.misc_kwargs)

                from map2map.models import lag2eul
                inp = d["input"][None]
                out = d["target"][None].clone()   # stand-in for the generator
                sty = d["style"][None]
                # train.py: input is resampled to the target grid before the cat
                if args.scale_factor != 1:
                    from map2map.models import resample
                    inp = resample(inp, args.scale_factor, narrow=False)
                from map2map.models import narrow_cast
                inp, out2, _ = narrow_cast(inp, out, out.clone())
                eul = lag2eul(out2[:, :3], a=float(sty))[0]
                fed_t = torch.cat([eul, out2], dim=1)
                if args.cgan:
                    fed_t = torch.cat([inp, fed_t], dim=1)
                fed = fed_t.shape[1]
                print(f"    D built from train.py's formula: in_chan={built}")
                print(f"    loop feeds {fed} channels "
                      f"(= cgan {inp.shape[1] if args.cgan else 0} + lag2eul "
                      f"{eul.shape[1]} + field {out2.shape[1]})")
                with torch.no_grad():
                    score = d_(fed_t, style=sty)
                print(f"    ok: {args.adv_model} FORWARD PASS -> "
                      f"{tuple(score.shape)}")

            # -- the extra loss term ------------------------------------
            # Constructed in gpu_worker, i.e. AFTER the allocation, so a bad
            # --extra-criterion-args or a broken import would otherwise be
            # discovered on a GPU. Built AND called here, on the real narrowed
            # shape, because construction is not the part that goes wrong.
            if getattr(args, "extra_criterion", None):
                ec = import_attr(args.extra_criterion, torch.nn, models,
                                 callback_at=args.callback_at)
                ec = ec(**args.extra_criterion_args)
                tgt_shape = (1, sum(ds.tgt_chan)) + (args.crop
                                                     * args.scale_factor,) * 3
                o = torch.zeros(tgt_shape, requires_grad=True)
                t = torch.zeros(tgt_shape)
                with torch.enable_grad():
                    v = args.extra_weight * ec(o, t, style=d["style"][None])
                print(f"    ok: {args.extra_criterion}"
                      f"({args.extra_criterion_args}) x{args.extra_weight} "
                      f"-> {float(v):.6g} on {tuple(tgt_shape)}")
                if not v.requires_grad:
                    failed.append("extra criterion is not differentiable")
                    print("    FAIL: the term returned a tensor with no "
                          "grad_fn, so it would\n          contribute nothing "
                          "to the generator and fail .backward().")

            # -- resume compatibility -----------------------------------
            # A checkpoint carries the SHAPES of whatever model wrote it. Change
            # a channel count between runs and the resume dies on a size
            # mismatch, minutes into the job and long after this stage used to
            # stop. That is how job 68237 failed: an earlier run had written D
            # with 21 input channels, the corrected code builds 20, and
            # load_state_dict refused.
            if args.load_state and Path(args.load_state).is_file():
                blob = torch.load(args.load_state, map_location="cpu",
                                  weights_only=False)
                for name, mod in (("model", m), ("adv_model", d_ if args.adv_model else None)):
                    if mod is None or name not in blob:
                        continue
                    missing, unexpected = mod.load_state_dict(blob[name],
                                                              strict=False)
                    bad = [k for k, v in blob[name].items()
                           if k in dict(mod.named_parameters())
                           and tuple(dict(mod.named_parameters())[k].shape)
                               != tuple(v.shape)]
                    if bad:
                        failed.append(f"{name} resume shape mismatch")
                        print(f"    FAIL: {args.load_state} cannot load into "
                              f"{name}: {len(bad)} shape mismatch(es), e.g.")
                        for k in bad[:3]:
                            print(f"      {k}: checkpoint "
                                  f"{tuple(blob[name][k].shape)} vs model "
                                  f"{tuple(dict(mod.named_parameters())[k].shape)}")
                        print( "          Move the checkpoint aside to start "
                               "clean, or restore the")
                        print( "          configuration that wrote it.")
                    else:
                        print(f"    ok: {args.load_state} loads into {name} "
                              f"({len(missing)} missing, {len(unexpected)} "
                              f"unexpected keys, 0 shape mismatches)")
        except Exception:
            failed.append("model/optimizer")
            traceback.print_exc()

    print()
    if failed:
        print(f"PREFLIGHT FAILED: {', '.join(failed)}")
        print("Fix these before spending a GPU allocation.")
        return 1
    print("PREFLIGHT PASSED -- their code accepted this configuration.")
    print("It does NOT say the run will produce a good field; the verdict is")
    print("density P/P_HR against frozen SR2's 0.977 (trilinear is 0.706).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
