"""Hourly accumulation and monthly output using Veros' native NetCDF serializer."""
from copy import copy
import json
import numpy as np
import h5netcdf
from veros.io_tools import netcdf
from veros.variables import TIMESTEPS

FIELDS = ('temp', 'salt', 'u', 'v', 'w', 'ssh', 'tke', 'eke')
STATIC = ('area_t', 'area_u', 'area_v', 'maskT', 'maskU', 'maskV', 'maskW',
          'dzt', 'dzw', 'ht', 'hu', 'hv', 'kbot', 'dxt', 'dxu', 'dyt', 'dyu', 'cost', 'cosu')

def sample(state):
    vs = state.variables
    fields = {}
    for name in FIELDS:
        if not state.var_meta[name].active:
            continue
        value = getattr(vs, name)
        if TIMESTEPS[0] in state.var_meta[name].dims:
            value = value[..., vs.tau]
        fields[name] = value
    for name in ('dtemp_nn', 'dsalt_nn', 'du_nn', 'dv_nn'):
        if name in state.var_meta and state.var_meta[name].active:
            fields[name] = getattr(vs, name)[..., vs.taum1]
    for name in ('u', 'v', 'w', 'dtemp_nn', 'dsalt_nn', 'du_nn', 'dv_nn'):
        if name not in fields:
            continue
        fields[name + '2'] = fields[name] ** 2
    return fields

class MonthlyOutput:
    def __init__(self, path, state, provenance, start_year=1993):
        self.start_year = start_year
        self.path = path
        self.metadata = {}
        with h5netcdf.File(path, 'w') as nc:
            netcdf.initialize_file(state, nc)
            nc.attrs.update(Conventions='CF-1.8', provenance=json.dumps(provenance),
                averaging='Arithmetic mean of hourly end-of-step samples; 720 samples per 30-day month',
                calendar_note=f'Initial condition placed at {start_year}-01-01; 360-day model calendar',
                forcing_note=provenance.get('forcing_note', 'Configured forcing; continuous trajectory without resets'),
                mld_note='Full-depth monthly T/S support offline density-threshold MLD; this differs from a monthly mean of instantaneous MLD',
                rho0=float(state.settings.rho_0), heat_capacity_reference=3991.86795711963)
            nc.variables['Time'].attrs.update(units=f'days since {start_year}-01-01 00:00:00', calendar='360_day', bounds='time_bounds', time_origin=f'01-JAN-{start_year} 00:00:00')
            nc.dimensions['bnds'] = 2
            nc.create_variable('time_bounds', ('Time', 'bnds'), 'f8')
            for name in ('simulation_year', 'simulation_month', 'sample_count'):
                nc.create_variable(name, ('Time',), 'i4')
            for name in STATIC:
                if name in nc.variables or name not in state.var_meta or not state.var_meta[name].active:
                    continue
                meta = copy(state.var_meta[name]); meta.time_dependent = False
                meta.get_mask = lambda settings, variables: None
                netcdf.initialize_variable(state, name, meta, nc)
                netcdf.write_variable(state, name, meta, np.asarray(getattr(state.variables, name)), nc)
            for name in sample(state):
                original = name[:-1] if name.endswith('2') else name
                meta = copy(state.var_meta[original])
                meta.dims = tuple(d for d in meta.dims if d not in TIMESTEPS)
                meta.time_dependent = True
                meta.dtype = 'float32'
                meta.extra_attributes = dict(meta.extra_attributes, cell_methods='Time: mean')
                if name.endswith('2'):
                    meta.name = 'Mean squared ' + original
                    meta.units = '(' + meta.units + ')2'
                self.metadata[name] = meta
                netcdf.initialize_variable(state, name, meta, nc)

    def write(self, state, sums, month_index, count=720):
        if count != 720:
            raise ValueError('Only complete 30-day months may be archived')
        with h5netcdf.File(self.path, 'a') as nc:
            if len(nc.dimensions['Time']) != month_index:
                raise ValueError('Monthly records must be sequential')
            netcdf.advance_time(month_index * 30 + 15, nc)
            nc.variables['time_bounds'][-1] = [month_index * 30, (month_index + 1) * 30]
            nc.variables['simulation_year'][-1] = self.start_year + month_index // 12
            nc.variables['simulation_month'][-1] = 1 + month_index % 12
            nc.variables['sample_count'][-1] = count
            for name, meta in self.metadata.items():
                netcdf.write_variable(state, name, meta, np.asarray(sums[name]) / count, nc)
