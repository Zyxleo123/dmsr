import numpy as np
OUT="/tmp/claude-1429/-zfsauton2-home-yixiz-DMSR/b94d5744-1d2e-484f-8b58-f18af32ff054/scratchpad"

def op_1d(N,w,g):
    m=N//w; H=np.zeros((m,N))
    for j in range(m):
        for k in range(w): H[j,(j*w+g+k)%N]=1.0/w
    return H
def G1(N,w,shifts):
    G=sum(op_1d(N,w,g).T@op_1d(N,w,g) for g in shifts)
    return np.linalg.eigvalsh(G)
def spec3(a,b,c):
    return np.sort((a[:,None,None]*b[None,:,None]*c[None,None,:]).ravel())[::-1]
def eff_rank(lam):
    lam=lam[lam>0]; p=lam/lam.sum(); return float(np.exp(-(p*np.log(p)).sum()))

thr_disp, thr_vel = 0.0083, 0.60
results={}
for (N,w,label) in [(64,8,"s8"),(64,2,"factor2")]:
    ef=G1(N,w,[0]); ea=G1(N,w,list(range(w)))
    single=ef.max()                       # single-operator scale (for eta, monotonic C2 bound)
    ef_s, ea_s = ef/single, ea/single     # single-op normalized (full >= fixed pointwise)
    cases={'fixed':(ef_s,ef_s,ef_s),'x':(ea_s,ef_s,ef_s),'xy':(ea_s,ea_s,ef_s),'xyz':(ea_s,ea_s,ea_s)}
    print("\n"+"="*76); print(f"### {label}  N={N} w={w}"); print("="*76)
    print(f"{'case':>6} | {'rank':>7} {'nullity':>8} {'effrank':>8} | nominal %>[1e-1 1e-2 1e-3 1e-4] | eta:disp  vel")
    R={}
    for name,(a,b,c) in cases.items():
        lam=spec3(a,b,c)
        lmax=lam.max(); lr_nom=lam/lmax                 # per-case (nominal table, plan's ask)
        lr_eta=lam                                       # single-op units already (>= across cases)
        rank=int((lam>1e-9*lmax).sum()); full=lam.size
        row=[int((lr_nom>t).sum()) for t in (1e-1,1e-2,1e-3,1e-4)]
        dsp=int((lr_eta>thr_disp).sum()); vel=int((lr_eta>thr_vel).sum())
        R[name]=dict(rank=rank,null=full-rank,eff=eff_rank(lam),nom=row,disp=dsp,vel=vel,full=full)
        print(f"{name:>6} | {rank:7d} {full-rank:8d} {eff_rank(lam):8.0f} | {row[0]:6d}{row[1]:6d}{row[2]:6d}{row[3]:6d}      | {dsp:6d} {vel:6d}")
    f,x=R['fixed'],R['xyz']
    print(f"  >>> null recovered {f['null']}->{x['null']} ({100*(1-x['null']/f['null']):.0f}%) | "
          f"DISP eta-modes {f['disp']}->{x['disp']} ({x['disp']/f['disp']:.1f}x) | "
          f"VEL eta-modes {f['vel']}->{x['vel']} ({x['vel']/f['vel']:.1f}x)")
    results[label]=R
    # save spectra for plotting
    np.savez(f"{OUT}/coverage_{label}.npz",
             **{f"{n}": spec3(*cases[n]) for n in cases})
np.save(f"{OUT}/coverage_results.npy", results, allow_pickle=True)
print("\nsaved spectra + results")
