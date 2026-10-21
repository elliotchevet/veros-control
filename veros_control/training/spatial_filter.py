"""Differentiable, conservative horizontal Helmholtz filter on wet neighbours.

Half-amplitude wavelength W corresponds to ell=W/(2*pi) in (I+ell^2 L)^-1.
No cross-depth coupling; coastlines have zero flux; longitude is periodic.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

DEFAULT_FILTER = dict(version='wet_helmholtz_v1', hours=[27.,81.,243.],
                      wavelengths_km=[0.,400.,800.], raw_fraction=0.1,
                      solver_tolerance=1e-8, solver_maxiter=256)


def validate_filter(config):
    hours=np.asarray(config['hours']); widths=np.asarray(config['wavelengths_km'])
    if config['version'] != 'wet_helmholtz_v1' or hours.ndim != 1 or len(hours)<2 or hours.shape != widths.shape:
        raise ValueError('Invalid spatial filter schedule')
    if not np.isfinite(hours).all() or not np.isfinite(widths).all() or np.any(np.diff(hours)<=0) or np.any(widths<0) or np.any(np.diff(widths)<0):
        raise ValueError('Filter schedule must be finite and increasing')
    if not 0 <= config['raw_fraction'] <= 1 or config['solver_tolerance']<=0 or config['solver_maxiter']<1:
        raise ValueError('Invalid filter mixture or solver settings')


def wavelength_at(hours, config):
    return jnp.interp(hours,jnp.asarray(config['hours']),jnp.asarray(config['wavelengths_km']))


def make_spatial_filter(variables, indices, config):
    validate_filter(config)
    wet=np.asarray(variables.maskT)[2:-2,2:-2,:][:,:,indices].astype(bool)
    area=np.asarray(variables.area_t)[2:-2,2:-2]/1e6  # km^2
    lon=np.unwrap(np.deg2rad(np.asarray(variables.xt)[2:-2]))
    lat=np.deg2rad(np.asarray(variables.yt)[2:-2])
    dx=6371.*np.diff(np.r_[lon,lon[0]+2*np.pi])[:,None]*np.cos(lat)[None,:]
    dy=np.broadcast_to(6371.*np.r_[np.diff(lat),lat[-1]-lat[-2]][None,:],area.shape)
    if np.any(area<=0) or np.any(dx<=0) or np.any(dy<=0):
        raise ValueError('Filter requires positive areas and increasing spherical coordinates')
    conductance=[]
    for axis,distance in enumerate((dx,dy)):
        other=np.roll(area,-1,axis=axis)
        face=2*area*other/(area+other)/distance**2
        edge=wet & np.roll(wet,-1,axis=axis)
        if axis==1:edge[:,-1,:]=False
        conductance.append(jnp.asarray(face[:,:,None]*edge))
    cx,cy=conductance
    mass=jnp.asarray(area); root=jnp.sqrt(mass)
    mask=jnp.asarray(wet)
    tol=config['solver_tolerance']

    def one_layer(field, wx, wy, ell2):
        def stiffness(f):
            fx=wx*(f-jnp.roll(f,-1,axis=0));fy=wy*(f-jnp.roll(f,-1,axis=1))
            return fx-jnp.roll(fx,1,axis=0)+fy-jnp.roll(fy,1,axis=1)
        def operator(q):return q+ell2*stiffness(q/root)/root
        diagonal=1+ell2*(wx+jnp.roll(wx,1,axis=0)+wy+jnp.roll(wy,1,axis=1))/mass
        rhs=root*field
        q,_=cg(operator,rhs,x0=rhs,tol=tol,atol=1e-12,maxiter=config['solver_maxiter'],M=lambda x:x/diagonal)
        residual=jnp.linalg.norm(operator(q)-rhs)
        ok=residual <= 10*(tol*jnp.linalg.norm(rhs)+1e-12)
        # Failed solves reject the training candidate instead of silently changing the loss.
        return jnp.where(ok,q/root,jnp.nan)

    def apply(error,hours):
        interior=jnp.where(mask,error[2:-2,2:-2,:],0.)
        width=wavelength_at(hours,config)
        def smooth(f):return jax.vmap(one_layer,in_axes=(2,2,2,None),out_axes=2)(f,cx,cy,(width/(2*jnp.pi))**2)
        filtered=jax.lax.cond(width>0,smooth,lambda f:f,interior)
        return jnp.pad(jnp.where(mask,filtered,0.),((2,2),(2,2),(0,0)))
    return apply
