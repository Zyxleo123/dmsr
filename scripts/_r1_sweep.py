"""Find an r1_gamma at which the HR critic actually discriminates.

Freezes the generator at the Stage B checkpoint (exactly the C/D initialisation),
trains ONLY the critic for a few hundred steps at each gamma, and reports:

  loss_D          -- 2.0 means the critic is at the uninformative fixed point
  separation      -- mean D(real) - D(fake); must become clearly positive
  grad_ratio      -- ||grad_adv|| / ||grad_flow|| at lambda_adv=0.1, the quantity
                     the design wants in 0.10-0.30

Run on a GPU node.
"""
import sys, time
sys.path.insert(0, "src")
import torch

from cosmo_sr.utils.config import load_config
from cosmo_sr.dmsr.data import resolve_split, build_paired_dataset
from cosmo_sr.data.datasets import infinite_loader
from cosmo_sr.dmsr.flow import build_flow, null_space_flow_loss
from cosmo_sr.dmsr.critic import HRCritic, LazyR1, hinge_d_loss, hinge_g_loss
from cosmo_sr.dmsr.density import CriticInputNormalizer, HighPassDensity, critic_input
from cosmo_sr.train import common

cfg = load_config("configs/dmsr/stage_c_critic_pairedlr.yaml")
d, ccfg, acfg = cfg["data"], cfg["critic"], cfg["adv"]
device = common.select_device(None)
ch = len(d["use_channels"])
factor = int(cfg["factor"])

split = resolve_split(d)
ds = build_paired_dataset(split, crop_lr=int(d["crop_lr"]), scale_factor=factor,
                          seed=0, augment=True, channels=int(d["channels"]),
                          use_channels=d["use_channels"], mmap=True)
loader = infinite_loader(ds, 2, seed=0)

flow = build_flow(cfg, ch).to(device)
blob = torch.load("runs/dmsr/stage_b/ckpt_best_frozen.pt", map_location=device, weights_only=False)
flow.load_state_dict(blob["model"])
for p in flow.parameters():
    p.requires_grad_(True)          # need grads for the ratio measurement
from cosmo_sr.dmsr.density import cellsizes
hr_cs, _ = cellsizes(d, factor)
hp = HighPassDensity(factor=factor, lowpass=ccfg["lowpass"],
                     cellsize=hr_cs, dis_norm=float(d["dis_norm"])).to(device)
print(f"HR cellsize={hr_cs:.4f} kpc/h")

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
GAMMAS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
ODE = int(acfg["gen_ode_steps"])

# Fixed batches so every gamma sees identical data.
batches = [common.to_device_batch(next(loader), device) for _ in range(STEPS)]
NORM = CriticInputNormalizer.fit([b["hr"] for b in batches[:16]], flow.operator, hp).to(device)
print(f"critic input scales: {NORM.to_dict()}")

print(f"steps={STEPS}  r1_interval={ccfg['r1_interval']}  lambda_adv(full)={acfg['lambda_adv']}")
print(f"{'gamma':>8}{'loss_D':>10}{'D(real)':>10}{'D(fake)':>10}{'separation':>12}"
      f"{'raw_R1':>10}{'grad_ratio':>12}  verdict")
print("-" * 92)

for gamma in GAMMAS:
    torch.manual_seed(0)
    critic = HRCritic(in_channels=ch + 1, width=int(ccfg["width"]),
                      n_layers=int(ccfg["n_layers"])).to(device)
    opt_d = torch.optim.Adam(critic.parameters(), lr=float(ccfg["lr"]),
                             betas=tuple(ccfg["betas"]))
    lazy = LazyR1(gamma=gamma, interval=int(ccfg["r1_interval"]))
    last_r1 = float("nan")

    for i in range(STEPS):
        b = batches[i]
        y, x = b["lr"], b["hr"]
        with torch.no_grad():
            x_fake = flow.generate(y, n_steps=ODE)
        real_in = critic_input(x, flow.operator, hp, normalizer=NORM).detach()
        fake_in = critic_input(x_fake, flow.operator, hp, normalizer=NORM).detach()
        s_real, s_fake = critic(real_in), critic(fake_in)
        loss = hinge_d_loss(s_real, s_fake)
        pen, m = lazy(critic, real_in)
        if pen is not None:
            loss = loss + pen
            last_r1 = m["loss_R1"]
        opt_d.zero_grad(set_to_none=True)
        loss.backward()
        opt_d.step()

    # final separation on a held-out-ish fresh batch
    b = batches[-1]
    y, x = b["lr"], b["hr"]
    with torch.no_grad():
        x_fake = flow.generate(y, n_steps=ODE)
        r = float(critic(critic_input(x, flow.operator, hp, normalizer=NORM)).mean())
        f = float(critic(critic_input(x_fake, flow.operator, hp, normalizer=NORM)).mean())
        ld = float(hinge_d_loss(critic(critic_input(x, flow.operator, hp, normalizer=NORM)),
                                critic(critic_input(x_fake, flow.operator, hp, normalizer=NORM))))

    # gradient ratio the generator would actually feel, at full lambda_adv
    lam = float(acfg["lambda_adv"])
    loss_flow, _ = null_space_flow_loss(flow, y, x)
    flow.zero_grad(set_to_none=True)
    loss_flow.backward()
    g_flow = common.grad_global_norm(list(flow.parameters()))
    x_fake = flow.generate(y, n_steps=ODE, bp_steps=1)
    adv = hinge_g_loss(critic(critic_input(x_fake, flow.operator, hp, normalizer=NORM)))
    flow.zero_grad(set_to_none=True)
    (lam * adv).backward()
    g_adv = common.grad_global_norm(list(flow.parameters()))
    flow.zero_grad(set_to_none=True)
    ratio = g_adv / max(g_flow, 1e-12)

    sep = r - f
    verdict = "INERT" if abs(sep) < 1e-3 else ("ok-sep" if sep > 0 else "INVERTED")
    if 0.10 <= ratio <= 0.30 and sep > 1e-3:
        verdict += " +RATIO-OK"
    print(f"{gamma:>8}{ld:>10.4f}{r:>10.5f}{f:>10.5f}{sep:>12.5f}"
          f"{last_r1:>10.3f}{ratio:>12.5f}  {verdict}")

print()
print("Target: separation clearly > 0 AND grad_ratio in 0.10-0.30.")
