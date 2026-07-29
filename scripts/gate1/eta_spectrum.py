import numpy as np, torch, torch.nn.functional as F, os
D="/zfsauton/scratch/yixiz/DMSR/paired_catnorm"
box="set15"  # held-out val box
hr = np.load(f"{D}/hr/{box}.npy", mmap_mode='r')   # (6,512,512,512)
lr = np.load(f"{D}/lr/{box}.npy")                    # (6,64,64,64)
C = hr.shape[0]; s = hr.shape[1]//lr.shape[1]
print(f"box={box}  HR{hr.shape}  LR{lr.shape}  s={s}")
lr_t = torch.from_numpy(lr.astype(np.float32))
def radial_ps(vol):  # vol: (N,N,N) -> (P(k), kbin)
    N=vol.shape[-1]; f=torch.fft.rfftn(vol); p=(f.real**2+f.imag**2)
    kx=torch.fft.fftfreq(N)*N; kz=torch.fft.rfftfreq(N)*N
    KX,KY,KZ=torch.meshgrid(kx,kx,kz,indexing='ij'); kk=torch.sqrt(KX**2+KY**2+KZ**2)
    kb=kk.round().long().flatten(); pf=p.flatten()
    nb=int(kb.max())+1; ps=torch.zeros(nb); cnt=torch.zeros(nb)
    ps.index_add_(0,kb,pf); cnt.index_add_(0,kb,torch.ones_like(pf))
    return (ps/cnt.clamp(min=1)), np.arange(nb)
print(f"\n{'ch':>3} {'var(LRsig)':>12} {'var(eta)':>12} {'eta_frac':>9}  {'hi-k eta_frac(k>0.5kNy)':>22}")
eta_frac=[]; eta_ps_list=[]; sig_ps_list=[]
for c in range(C):
    Ac = F.avg_pool3d(torch.from_numpy(hr[c].astype(np.float32))[None,None], s, s)[0,0]  # (64,64,64)
    eta = lr_t[c] - Ac
    vsig=Ac.var().item(); veta=eta.var().item(); frac=veta/vsig
    ps_e,kb = radial_ps(eta); ps_s,_ = radial_ps(Ac)
    kNy=len(kb)-1; hi = kb> 0.5*kNy
    hi_frac = ps_e[hi].sum().item()/ (ps_e.sum().item()+1e-30)
    eta_frac.append(frac); eta_ps_list.append(ps_e.numpy()); sig_ps_list.append(ps_s.numpy())
    print(f"{c:>3} {vsig:12.5f} {veta:12.5f} {frac:9.4f}  {hi_frac:22.3f}")
np.savez("/tmp/claude-1429/-zfsauton2-home-yixiz-DMSR/b94d5744-1d2e-484f-8b58-f18af32ff054/scratchpad/eta_spectrum.npz",
         eta_frac=np.array(eta_frac), eta_ps=np.array(eta_ps_list), sig_ps=np.array(sig_ps_list))
print("\ndisp mean eta_frac:", np.mean(eta_frac[:3]), " vel mean eta_frac:", np.mean(eta_frac[3:]))
