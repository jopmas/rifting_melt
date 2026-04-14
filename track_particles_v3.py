import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, glob, gc
import xarray as xr
import sys

cdir = sys.argv[-2]
os.chdir(cdir)
path_param = os.path.join(cdir,'param.txt')

def read_params(path_param):
    '''
    Carrega os parâmetros do arquivos param.txt em forma de dicionario
    '''
    params_form = {}
    ptemp = ''
    with open(path_param,'r') as param:
        for line in param:
            line = line.strip()
            if len(line)==0:
                continue
            elif line[0] == "#":
                continue
            
            line = line.split('#')[0]
            line = line.replace(' ','')
            pv = line.split('=')
            params_form[pv[0].lower()] = pv[1]
            ptemp = ptemp + line+'\n'

    return params_form

def get_rank(cdir):
    from pathlib import Path
    return int(len(list(Path(f'{cdir}/steps').glob("step_0_*"))))


def load_particles(step, filtering=None,filters_cond=[]):
    '''
    read all files from a given step and return a dataframe with all particles exported at this step
    filter by loc = [selection_on_X,selection_on_Z,selection_by_layer]
    filter by id = [particles_IDs]
    
    '''
    ranks = get_rank(cdir)
    particles_step = []
    for rank in range(ranks):
        #x1,z1,id,layer,epsilom = np.loadtxt("step_"+str(step_final)+"_"+str(rank)+".txt",unpack=True)
        #print(rank)
        file = open(f"steps/step_{step}_{rank}.txt",'r')
        A = file.read().split('\n')[:-1]
        for l in A:
            p = l.split(' ')
            particles_step.append(p[:-1])
        file.close()
        
    ps = pd.DataFrame(particles_step, columns=['x', 'z', 'id', 'layer']).astype(float).astype(int)
    if filtering:
        if filtering=='pos':
            print("Filtering by position")
            cond = (ps.z >= filters_cond[1][0]) & \
                   (ps.z <= filters_cond[1][1]) & \
                   (np.isin(ps.layer, filters_cond[2])) & \
                   (ps.x >= filters_cond[0][0]) & \
                   (ps.x <= filters_cond[0][1])
                   
        elif filtering=='id':
            print("filtering by id list")
            cond = np.isin(ps.id, filters_cond)
        ps = ps.loc[cond]
    return ps

params = read_params(path_param)
Nx = int(params['nx'])
Nz = int(params['nz'])
Lx = int(float(params['lx'])) #m
Lz = int(float(params['lz'])) #m
timespd = pd.read_csv(f"{cdir}/times.csv")
tair = 40e3 #km - air thickness
id_wks = 2 #weak seed id

step_initial = timespd.iloc[0,0]
step_final = timespd.iloc[-2,0]
cores = get_rank(cdir)
print(f"Final step: {step_final}")
print("Number of ranks:", cores)

#selection of particles
step_selection = int(sys.argv[-1])
part_selec_layer = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] #particles selection by material
part_selec_xpos = [0, Lx] #particles selection by x position
part_selec_zpos = [-Lz, 0.0e3] #particles selection by z position
filtering_conds = [part_selec_xpos, part_selec_zpos, part_selec_layer]

ps = load_particles(step_selection,filtering='pos',filters_cond=filtering_conds)
ps['time'] = timespd[timespd.step==step_selection].iloc[0,1]

#testing selection
plt.figure('teste selection', figsize=(16,4))
plt.text(750,-1,
         f"{ps.time.iloc[0]} myr",
         bbox=dict(facecolor='white', alpha=0.5))
plt.scatter(ps.x/1e3,(ps.z+tair)/1e3,c=ps.layer,s=0.5,cmap='Paired',vmin=-0.5,vmax=14.5)
plt.colorbar()
plt.xlim(np.array(part_selec_xpos)/1e3)
plt.tight_layout()
plt.savefig('particles_selection.png',dpi=150)

id_array = np.array(ps.id) #id of the particles to track

dfs = []
for step in list(timespd.iloc[:,0]):
    tmy = timespd[timespd.step==step].iloc[0,1]
    print(f"processing step {step} ({tmy} myr)")
    pstep = load_particles(step,filtering='id',filters_cond=id_array)
    pstep['time'] = tmy
    dfs.append(pstep)
    
    #pstep.z = pstep.z+tair #correction for z axis

#creating the xarray
particles_total = pd.concat(dfs)
particles_total.z = particles_total.z+Lz #necessary to fit the grid coordinate format

if id_wks > 0:
    particles_total.loc[(particles_total.layer>id_wks-1) & (particles_total.layer<=id_wks+1),'layer'] = id_wks-1 #fixing layer IDs due to mantle weak seed
    particles_total.loc[(particles_total.layer>id_wks+1),'layer'] -= id_wks
del dfs[:]; del dfs; gc.collect()

df_indexed = particles_total.set_index(['id','time'])
df_indexed['layer']=df_indexed['layer'].astype("int8")
ds = df_indexed.to_xarray()

ds.attrs = {'description': 'particle trajectories', 
            'reference timestep': f'{ps.time.iloc[0]}myr ({step_selection})',
            'selected layers' : str(part_selec_layer),
            'disclaimers':'some particles can change its layer over time (sedimentation/erosion)'}

if id_wks > 0:
    ds.attrs['weak seed'] = str(id_wks)

#

ds['time'].attrs = {'units': 'myr', 'long_name': 'time'}
ds['x'].attrs = {'units': 'm', 'long_name': 'x coordinates'}
ds['z'].attrs = {'units': 'm', 'long_name': 'z coordinates'}

comp = dict(zlib=True, complevel=5, shuffle=True)
encoding_settings={
    'x': {**comp, 'dtype': 'int32','_FillValue':-999}, 
    'z': {**comp, 'dtype': 'int32','_FillValue':-999},
    
    # Variáveis Inteiras (ID, Layer): Garantir que são int e não float
    'id':    {**comp, 'dtype': 'int64'}, #int32: -2147483648:2147483647
    'layer': {**comp, 'dtype': 'int8', '_FillValue': -1},  # int8: -128-127
}

ds.to_netcdf("particles_trajectories.nc",encoding=encoding_settings,engine='netcdf4')

