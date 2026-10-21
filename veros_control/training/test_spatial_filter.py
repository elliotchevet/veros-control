"""Small-grid independent checks of the filter and its implicit derivative."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import jax
import jax.numpy as jnp
spec=importlib.util.spec_from_file_location('spatial_filter',Path(__file__).with_name('spatial_filter.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SpatialFilterTests(unittest.TestCase):
    def grid(self):
        nx,ny,nz=8,4,2
        lon=np.arange(nx)*360/nx;lat=np.arange(ny)*2-3
        vs=NS(xt=np.r_[lon[-2:],lon,lon[:2]],yt=np.r_[lat[:2]-4,lat,lat[-2:]+4],
              maskT=np.ones((nx+4,ny+4,nz)),area_t=np.broadcast_to((100+np.arange(ny+4))[None,:],(nx+4,ny+4))*1e6)
        return vs

    def test_schedule_identity_and_constant(self):
        cfg=dict(m.DEFAULT_FILTER)
        np.testing.assert_allclose(m.wavelength_at(jnp.array([0.,27.,54.,81.,162.,243.]),cfg),[0,0,200,400,600,800])
        v=self.grid();f=m.make_spatial_filter(v,np.arange(2),cfg)
        x=jnp.ones(v.maskT.shape)*3
        for t in [0.,27.,243.]:np.testing.assert_allclose(f(x,t)[2:-2,2:-2],3.,rtol=1e-8)
        rng=np.random.default_rng(0);x=jnp.array(rng.normal(size=v.maskT.shape))
        np.testing.assert_array_equal(f(x,27.)[2:-2,2:-2],x[2:-2,2:-2])

    def test_dense_reference_conservation_and_gradient(self):
        v=self.grid();cfg=dict(m.DEFAULT_FILTER);f=m.make_spatial_filter(v,np.arange(2),cfg)
        x=np.random.default_rng(2).normal(size=v.maskT.shape)
        y=np.asarray(f(jnp.asarray(x),243.));area=v.area_t[2:-2,2:-2]/1e6
        nx,ny=area.shape;K=np.zeros((nx*ny,nx*ny));ell2=(800/(2*np.pi))**2
        for i in range(nx):
            for j in range(ny):
                for ii,jj,d in [((i+1)%nx,j,6371*np.deg2rad(45)*np.cos(np.deg2rad(v.yt[j+2]))),(i,j+1,6371*np.deg2rad(2))]:
                    if jj>=ny:continue
                    a=i*ny+j;b=ii*ny+jj;c=2*area[i,j]*area[ii,jj]/(area[i,j]+area[ii,jj])/d**2
                    K[a,a]+=c;K[b,b]+=c;K[a,b]-=c;K[b,a]-=c
        A=np.diag(area.ravel())+ell2*K
        expected=np.linalg.solve(A,area.ravel()*x[2:-2,2:-2,0].ravel()).reshape(nx,ny)
        np.testing.assert_allclose(y[2:-2,2:-2,0],expected,atol=1e-7)
        np.testing.assert_allclose(np.sum(y[2:-2,2:-2]*area[:,:,None],axis=(0,1)),np.sum(x[2:-2,2:-2]*area[:,:,None],axis=(0,1)),rtol=1e-7)
        loss=lambda a:jnp.sum(f(a,243.)**2)
        grad=jax.grad(loss)(jnp.asarray(x));direction=np.random.default_rng(3).normal(size=x.shape)
        fd=float((loss(x+1e-4*direction)-loss(x-1e-4*direction))/2e-4)
        self.assertAlmostEqual(float(jnp.sum(grad*direction)),fd,places=5)

    def test_land_barriers_and_vertical_independence(self):
        v=self.grid();v.maskT[2,:,:]=0;v.maskT[6,:,:]=0
        f=m.make_spatial_filter(v,np.arange(2),dict(m.DEFAULT_FILTER))
        x=np.zeros(v.maskT.shape);x[3:6,2:-2,0]=2.;x[7:10,2:-2,0]=5.
        x[2,:,:]=np.nan;x[6,:,:]=np.nan
        y=np.asarray(f(jnp.asarray(x),243.))
        np.testing.assert_allclose(y[3:6,2:-2,0],2.,rtol=1e-8)
        np.testing.assert_allclose(y[7:10,2:-2,0],5.,rtol=1e-8)
        np.testing.assert_allclose(y[:,:,1],0.)
        self.assertTrue(np.isfinite(y).all())

    def test_filtered_basin_loss_and_depth_calibration(self):
        from veros_control.training.basin_loss import make_basin_loss, basin_masks
        v=self.grid();v.zt=np.array([-200.,-5.]);v.dzt=np.array([100.,10.])
        config=dict(m.DEFAULT_FILTER)
        initial={k:np.zeros(v.maskT.shape) for k in ('temp','salt')}
        target={k:np.ones(v.maskT.shape) for k in ('temp','salt')}
        field=jnp.asarray(np.random.default_rng(5).normal(size=v.maskT.shape))
        _,_,terms,meta=make_basin_loss(v,target,levels=0,pattern_lambda=1.,initial_target=initial,
            target_alpha=lambda s:s.variables.time/720.,spatial_filter=config,
            lead_hours=lambda s:s.variables.time,return_depth_errors=True)
        indices=np.asarray(meta['level_indices'])
        masks=basin_masks(v.xt,v.yt)
        masks[:,:2,:]=False;masks[:,-2:,:]=False;masks[:,:,:2]=False;masks[:,:,-2:]=False
        filter_error=m.make_spatial_filter(v,indices,config)
        for hour in [27.,243.]:
            error=field[:,:,indices]-hour/720.
            smooth=np.asarray(filter_error(error,hour))
            expected=[]
            for mask in masks:
                weights=mask*v.area_t
                if weights.sum()>0:
                    expected.append(np.sum(weights[:,:,None]*(.1*np.asarray(error)**2+.9*smooth**2),axis=(0,1))/weights.sum())
            state=NS(variables=NS(temp=field,salt=field,time=hour,tau=0))
            np.testing.assert_allclose(terms(state),np.broadcast_to(np.mean(expected,axis=0),(2,2)),rtol=1e-7)

if __name__=='__main__':unittest.main()
