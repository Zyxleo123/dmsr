"""Progressive unfreezing of the pretrained SR2 generator, by exact name.

Fine-tuning a checkpoint that took days of adversarial training to produce, on a
reward whose gradient comes through a *learned surrogate*, is an invitation to
destroy it in a few hundred steps. The mitigation this module implements is
neither "small learning rate" nor "early stopping": it is **restricting which
parameters exist to the optimiser at all**, and doing so by exact name so the
set is auditable rather than inferred from a substring match.

The rungs, cheapest first
-------------------------
``proj_noise``
    ``blocks.*.proj.*`` and the per-channel noise scales
    ``blocks.*.conv.0.std`` / ``blocks.*.conv.4.std``. This is the smallest set
    that can change the *output* at all without touching a single 3x3x3 kernel:
    ``proj`` is the 1x1 head that writes into the skip sum (i.e. into the
    displacement/velocity field directly), and ``std`` is the amplitude the
    generator gives its own stochasticity. On ``G_z0`` the ``std`` values are
    ~1e-3 at z0..z4 and ~5e-2 at z5, so this rung is overwhelmingly a control on
    fine-scale stochastic structure -- which is what the occupation failure is
    about.
``fine``, ``middle_fine``, ``all_blocks``
    Add the convolution weights of ``blocks.2``, then ``blocks.1``, then
    ``blocks.0``. Note the order: block 2 acts at 4x-8x resolution and block 0 at
    the LR resolution, so this walks from the scales SR2 gets wrong toward the
    scales it gets right.
``full``
    Adds ``block0``, the 1x1 input embedding. Last, because a change there is
    seen by every subsequent layer at every scale.

A rung is a *stage*, not a schedule
-----------------------------------
Each rung is run as its own checkpointed job and is entered only when the
previous one improved the **real** catalog reward without breaking a field
guard. Unfreezing more is never the response to a proxy that fails its gate;
that is documented in ``configs/reward/sr2_direct_finetune.yaml`` and enforced
in the submitter's stage ordering, not here.

Learning rates
--------------
Parameters are grouped by *depth*, not by rung, so a group's rate does not
change when a later rung is entered: projection/noise, fine convolutions, middle
convolutions, and coarse/input layers. The defaults
(:data:`DEFAULT_GROUP_LR`) span a factor of ~30 from the shallowest to the
deepest group, and every one of them is configurable.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "DEFAULT_GROUP_LR",
    "GROUP_NAMES",
    "RUNG_ORDER",
    "assert_only_trainable_changed",
    "describe_trainable",
    "group_of_parameter",
    "next_rung",
    "print_trainable",
    "rung_names",
    "parameter_groups",
    "set_trainable",
    "snapshot_parameters",
    "trainable_names",
]

#: Rung names, cheapest first. The order is the escalation order.
RUNG_ORDER: Tuple[str, ...] = (
    "proj_noise", "fine", "middle_fine", "all_blocks", "full",
)

#: Learning-rate groups, shallow (closest to the output) first.
GROUP_NAMES: Tuple[str, ...] = ("proj_noise", "fine", "middle", "coarse")

#: Suggested initial rates. Every value is overridable from the config.
DEFAULT_GROUP_LR: Dict[str, float] = {
    "proj_noise": 1e-5,
    "fine": 3e-6,
    "middle": 1e-6,
    "coarse": 3e-7,
}

# Which ``blocks.<b>`` index each depth group owns. ``block0`` (the input
# embedding) is coarse. Indices are for the standard ``scale_factor = 8``
# generator, which has exactly three blocks.
_CONV_SUFFIXES = ("conv.2.weight", "conv.2.bias", "conv.5.weight", "conv.5.bias")
_NOISE_SUFFIXES = ("conv.0.std", "conv.4.std")


def _block_indices(model: nn.Module) -> List[int]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError(
            f"{type(model).__name__} has no `blocks`; this utility is for "
            "cosmo_sr.tts.srs_noise.ControlledG"
        )
    return list(range(len(blocks)))


def _proj_and_noise_names(model: nn.Module) -> List[str]:
    out: List[str] = []
    for b in _block_indices(model):
        out += [f"blocks.{b}.proj.0.weight", f"blocks.{b}.proj.0.bias"]
        out += [f"blocks.{b}.{s}" for s in _NOISE_SUFFIXES]
    return out


def _block_conv_names(block: int) -> List[str]:
    return [f"blocks.{block}.{s}" for s in _CONV_SUFFIXES]


def _block0_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_parameters() if n.startswith("block0.")]


def rung_names(model: nn.Module, rung: str) -> List[str]:
    """The exact parameter names a rung makes trainable.

    Raises on a name the model does not have, so a rung definition can never
    silently unfreeze *fewer* parameters than it claims (which would look like a
    rung that simply did not help).
    """
    rung = str(rung)
    if rung not in RUNG_ORDER:
        raise ValueError(f"unknown rung {rung!r}; expected one of {list(RUNG_ORDER)}")
    blocks = _block_indices(model)
    if len(blocks) != 3:
        raise ValueError(
            f"the rung definitions name blocks 0/1/2 explicitly, but this "
            f"generator has {len(blocks)} blocks; define rungs for it before use"
        )
    names = list(_proj_and_noise_names(model))
    if RUNG_ORDER.index(rung) >= RUNG_ORDER.index("fine"):
        names += _block_conv_names(2)
    if RUNG_ORDER.index(rung) >= RUNG_ORDER.index("middle_fine"):
        names += _block_conv_names(1)
    if RUNG_ORDER.index(rung) >= RUNG_ORDER.index("all_blocks"):
        names += _block_conv_names(0)
    if rung == "full":
        names += _block0_names(model)

    have = {n for n, _ in model.named_parameters()}
    missing = [n for n in names if n not in have]
    if missing:
        raise KeyError(
            f"rung {rung!r} names parameters this generator does not have: "
            f"{missing[:8]}"
        )
    # Deterministic and duplicate-free: the printed list is evidence, and a
    # duplicate would double a parameter's weight decay in the optimiser.
    return sorted(dict.fromkeys(names))


def group_of_parameter(name: str) -> str:
    """Which learning-rate group a parameter name belongs to.

    Depth-based, and independent of the rung: ``blocks.2.conv.2.weight`` is a
    ``fine`` parameter whether it was unfrozen at rung ``fine`` or at rung
    ``full``.
    """
    n = str(name)
    if n.endswith(".std") or ".proj." in n:
        return "proj_noise"
    if n.startswith("block0."):
        return "coarse"
    if n.startswith("blocks.2."):
        return "fine"
    if n.startswith("blocks.1."):
        return "middle"
    if n.startswith("blocks.0."):
        return "coarse"
    raise KeyError(f"no learning-rate group for parameter {name!r}")


def set_trainable(model: nn.Module, rung: str) -> List[str]:
    """Freeze everything, then unfreeze exactly the rung. Returns the names.

    The freeze-first step is not defensive padding: a stage that resumes from
    the previous rung's checkpoint inherits its ``requires_grad`` flags, so
    without it "rung 2" would mean "rung 1 plus rung 2" only by luck of ordering
    and "rung 1 minus something" if the stages were ever run out of order.
    """
    names = rung_names(model, rung)
    wanted = set(names)
    for n, p in model.named_parameters():
        p.requires_grad_(n in wanted)
    return names


def trainable_names(model: nn.Module) -> List[str]:
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def parameter_groups(
    model: nn.Module,
    rung: str,
    lrs: Optional[Mapping[str, float]] = None,
    *,
    weight_decay: float = 0.0,
) -> List[Dict]:
    """Optimiser param groups for a rung, one group per depth.

    Empty groups are dropped: several optimisers accept them, but an empty group
    in a logged config reads as "this depth was being trained at 3e-7" when in
    fact it was frozen.
    """
    rates = dict(DEFAULT_GROUP_LR)
    for k, v in dict(lrs or {}).items():
        if k not in rates:
            raise KeyError(
                f"unknown learning-rate group {k!r}; expected {list(GROUP_NAMES)}"
            )
        rates[k] = float(v)

    names = set_trainable(model, rung)
    by_name = dict(model.named_parameters())
    buckets: Dict[str, List[torch.nn.Parameter]] = {g: [] for g in GROUP_NAMES}
    for n in names:
        buckets[group_of_parameter(n)].append(by_name[n])
    return [
        {"params": buckets[g], "lr": float(rates[g]), "name": g,
         "weight_decay": float(weight_decay)}
        for g in GROUP_NAMES if buckets[g]
    ]


def describe_trainable(
    model: nn.Module,
    rung: str,
    lrs: Optional[Mapping[str, float]] = None,
) -> Dict:
    """The complete trainable-parameter list, count and per-group breakdown.

    Returned *and* printable. Printing the full list rather than a count is
    deliberate: the entire safety argument for this line is "only these tensors
    move", and a count cannot be checked against that claim by a reader.
    """
    rates = {**DEFAULT_GROUP_LR, **dict(lrs or {})}
    names = rung_names(model, rung)
    by_name = dict(model.named_parameters())
    rows = []
    for n in names:
        p = by_name[n]
        rows.append({
            "name": n,
            "group": group_of_parameter(n),
            "shape": list(p.shape),
            "numel": int(p.numel()),
            "lr": float(rates[group_of_parameter(n)]),
        })
    total_all = sum(p.numel() for p in model.parameters())
    per_group: Dict[str, Dict[str, float]] = {}
    for g in GROUP_NAMES:
        sel = [r for r in rows if r["group"] == g]
        if sel:
            per_group[g] = {"n_tensors": len(sel),
                            "n_params": int(sum(r["numel"] for r in sel)),
                            "lr": float(rates[g])}
    return {
        "rung": str(rung),
        "n_trainable_tensors": len(rows),
        "n_trainable_params": int(sum(r["numel"] for r in rows)),
        "n_total_params": int(total_all),
        "trainable_fraction": float(sum(r["numel"] for r in rows)) / max(total_all, 1),
        "per_group": per_group,
        "parameters": rows,
    }


def print_trainable(model: nn.Module, rung: str,
                    lrs: Optional[Mapping[str, float]] = None) -> Dict:
    d = describe_trainable(model, rung, lrs)
    print(f"=== unfreezing rung {d['rung']!r}: {d['n_trainable_tensors']} tensors, "
          f"{d['n_trainable_params']} / {d['n_total_params']} parameters "
          f"({100.0 * d['trainable_fraction']:.3f}%)", flush=True)
    for row in d["parameters"]:
        print(f"    {row['name']:<32s} {str(row['shape']):<24s} "
              f"group={row['group']:<10s} lr={row['lr']:.2e}", flush=True)
    for g, s in d["per_group"].items():
        print(f"    [{g}] {s['n_tensors']} tensors, {s['n_params']} params, "
              f"lr {s['lr']:.2e}", flush=True)
    return d


def next_rung(rung: str) -> Optional[str]:
    """The rung after this one, or ``None`` at the top."""
    i = RUNG_ORDER.index(str(rung))
    return RUNG_ORDER[i + 1] if i + 1 < len(RUNG_ORDER) else None


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def snapshot_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Detached CPU copies of every parameter, for a before/after comparison."""
    return {n: p.detach().to("cpu").clone() for n, p in model.named_parameters()}


def assert_only_trainable_changed(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
    expected: Iterable[str],
    *,
    atol: float = 0.0,
) -> Dict[str, float]:
    """Raise if a parameter outside ``expected`` moved. Returns per-tensor deltas.

    ``atol = 0`` is the right default and not a strict one: a parameter with
    ``requires_grad = False`` receives no gradient, so its post-step value is
    bit-identical unless something (weight decay on a stale group, an EMA write-back,
    a second optimiser) touched it. Any nonzero delta there is a bug, not noise.
    """
    exp = {str(e) for e in expected}
    deltas: Dict[str, float] = {}
    offenders: List[str] = []
    for n, p in model.named_parameters():
        if n not in before:
            offenders.append(f"{n} (absent from the snapshot)")
            continue
        d = float((p.detach().to("cpu") - before[n]).abs().max())
        deltas[n] = d
        if n not in exp and d > float(atol):
            offenders.append(f"{n} (|delta|max={d:.3e})")
    if offenders:
        raise AssertionError(
            "frozen parameters changed after an optimizer step: "
            + ", ".join(offenders[:10])
        )
    return deltas
