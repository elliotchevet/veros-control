"""Area-weighted horizontal errors, reported separately at every depth."""
import numpy as np


def depth_errors(field,target,area,wet):
    wet=np.array(wet,dtype=bool)
    wet[:2]=False;wet[-2:]=False;wet[:,:2]=False;wet[:,-2:]=False
    weights=np.where(wet,area[...,None],0.)
    total=weights.sum(axis=(0,1))
    error=np.where(wet,field-target,0.)
    denominator=np.where(total>0,total,np.nan)
    return dict(bias=(weights*error).sum(axis=(0,1))/denominator,
        rmse=np.sqrt((weights*error**2).sum(axis=(0,1))/denominator))
