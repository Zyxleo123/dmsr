import sys, torch, numpy as np
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from cosmo_sr.utils.config import load_config
from dmsr_eval import load_flow
from cosmo_sr.data.field_io import load_field
dev=torch.device('cuda')
lr=load_field('/zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy').astype('float32')
for lbl,cfgp,ck in [('paired_det','configs/dmsr/paired_deterministic.yaml','runs/dmsr/paired_deterministic/ckpt_best.pt'),
                    ('stage_e','configs/dmsr/mean_innovation_e.yaml','runs/dmsr/mean_innovation_e_s0/ckpt_best.pt')]:
    cfg=load_config(cfgp); uc=cfg.get('data',{}).get('use_channels') or [0,1,2]
    m=load_flow(cfg,len(uc),ck,dev,use_ema=True)
    y=torch.from_numpy(np.ascontiguousarray(lr[uc]))[None].to(dev)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad(): x=m.generate(y,n_steps=20)
        print(f'{lbl}: OK full-box {tuple(x.shape)} peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB',flush=True)
        del x
    except RuntimeError as e:
        print(f'{lbl}: OOM/ERR {str(e)[:120]}',flush=True)
    del m; torch.cuda.empty_cache()
