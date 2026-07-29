import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S="/tmp/claude-1429/-zfsauton2-home-yixiz-DMSR/b94d5744-1d2e-484f-8b58-f18af32ff054/scratchpad"
fig,axes=plt.subplots(1,2,figsize=(13,5))
titles={'s8':'s=8 single-jump (64→512), w=8','factor2':'factor-2 octave (32→64), w=2'}
colors={'fixed':'#444','x':'#1b9e77','xy':'#d95f02','xyz':'#7570b3'}
labels={'fixed':'fixed A','x':'x shifts','xy':'xy shifts','xyz':'xyz shifts'}
for ax,key in zip(axes,['s8','factor2']):
    d=np.load(f"{S}/coverage_{key}.npz")
    for name in ['fixed','x','xy','xyz']:
        lam=np.sort(d[name])[::-1]; lam=lam/lam.max()
        ax.semilogy(np.arange(lam.size)/lam.size, np.clip(lam,1e-6,None),
                    color=colors[name], lw=1.8, label=labels[name])
    ax.axhline(0.0083, color='tab:blue', ls='--', lw=1, alpha=.7)
    ax.axhline(0.60, color='tab:red', ls='--', lw=1, alpha=.7)
    ax.text(0.02,0.0083*1.3,'disp η-floor 0.008',color='tab:blue',fontsize=8)
    ax.text(0.02,0.60*1.15,'vel η-floor 0.60',color='tab:red',fontsize=8)
    ax.set_title(titles[key]); ax.set_xlabel('fraction of modes (sorted)'); ax.set_ylim(1e-6,2)
    ax.set_ylabel('eigenvalue of G / max'); ax.legend(fontsize=9); ax.grid(alpha=.3)
fig.suptitle('Gate 1: operator coverage of $G=\\sum_g H_g^T H_g$  (subcell-shift diversity vs fixed A)',fontsize=12)
fig.tight_layout(); fig.savefig("docs/figures/gate1_operator_coverage.png",dpi=130)
print("saved docs/figures/gate1_operator_coverage.png")
