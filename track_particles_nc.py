########################################################################
# BEFORE RUNNING THIS SCRIPT:                                          #
#  - CREATE THE .nc FILES FROM .txt MANDYOC OUTPUTS                    #
#  - CREATE THE _steps.nc FILE WITH StepsNETCDF.jl                     #
#  - RUN THIS SCRIPT IN THE SAME FOLDER WHERE THE .nc FILES ARE        #
#                                                                      #
# Same as track_particles.py, but reads the particles from _steps.nc   #
# instead of the steps/step*.txt files                                 #
########################################################################

import numpy as np
import os
import xarray as xr
import subprocess
import multiprocessing #needed to run pymp in macOS
multiprocessing.set_start_method('fork') #needed to run pymp in macOS arm architectures
import pymp

model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]

#reading datasets
Tdataset = xr.open_dataset("_temperature.nc")
Pdataset = xr.open_dataset("_pressure.nc")
Sdataset = xr.open_dataset("_steps.nc")

#getting the dimensions
Nx = int(Tdataset.nx)
Nz = int(Tdataset.nz)
Lx = float(Tdataset.lx)
Lz = float(Tdataset.lz)

print(
    "nx:", Nx, "\n",
    "nz:", Nz, "\n",
    "Lx:", Lx, "\n",
    "Lz:", Lz
)

xi = np.linspace(0,Lx,Nx)
zi = np.linspace(-Lz,0,Nz)
xx,zz = np.meshgrid(xi,zi)

dx = Lx/(Nx-1)
dz = Lz/(Nz-1)

if('He' in model_name):
    take_asthenosphere = True
else:
    take_asthenosphere = False

#select the extension rate:
if('W1' in model_name or 'S1' in model_name):
    v=1 #cm/yr #For groups 1,2 and 3
if('W2' in model_name or 'S2' in model_name):
    v=2 #cm/yr #For groups 4,5 and 6
if('W3' in model_name or 'S3' in model_name):
    v=3 #cm/yr #For groups 7,8 and 9

if(v==1):
    instant_to_take = 40 #Myr
if(v==2):
    instant_to_take = 40#20 #Myr
if(v==3):
    instant_to_take = 40#8.55 #Myr

if(Tdataset.time.values[-1] < instant_to_take):
    print(f"WARNING: the model only reaches {Tdataset.time.values[-1]:.2f} Myr (< {instant_to_take} Myr). The last step will be used.")

idx = (np.abs(Tdataset.time.values - instant_to_take)).argmin()
step_final = Tdataset.steps.values[idx]

print(f"Track particles until {instant_to_take} Myr, idx: {idx}, time idx: {Tdataset.time.values[idx]} Myr")

print("Final step: ", step_final)

#index of each step of Tdataset inside _steps.nc
steps_nc = Sdataset.steps.values
missing_steps = np.setdiff1d(Tdataset.steps.values[:idx+1], steps_nc)
if(missing_steps.size > 0):
    raise ValueError(f"Steps not found in _steps.nc: {missing_steps}")

def steps_nc_index(step):
    return int(np.where(steps_nc == step)[0][0])

#Reading the data of final step
final = Sdataset.isel(time=steps_nc_index(step_final))
exists = np.isfinite(final.x.values)
x = final.x.values[exists]
z = final.z.values[exists]
id_vec = final.particle.values[exists]
layer_vec = final.layer.values[exists]


###########################################################################
#Selecting the particles to track                                         #
# All particles above search_thickness and between x_begin and x_end [km] #
# and does not belong to air                                              #
###########################################################################

h_air = 40.0e3

#Determining the thickness to search for lithospheric mantle particles
search_thickness = 12.0e3

x_begin = 600.0e3
x_end = 1600.0e3

asthenosphere_code = 0
mantle_lithosphere1_code = 1
seed_code = 2
mantle_lithosphere2_code = 3
lower_crust_code = 4
upper_crust_code = 5
air_code = 6

print(f"Selecting particles between {x_begin/1000} and {x_end/1000} km, above {search_thickness/1000} km")

if(take_asthenosphere):
    cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<upper_crust_code) & (layer_vec>=asthenosphere_code) #to take particles from asthenosphere and lithosphere.
else:
    cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<upper_crust_code) & (layer_vec>asthenosphere_code) #to take particles from lithosphere

part_selec = id_vec[cond]
layers_selec = layer_vec[cond] #get the layer numbers of the selected particles
n_tracked = np.size(part_selec)

print(f"x size: {np.size(x)}")
print(f"Number of tracked particles (n): {np.size(part_selec)}, where:")

n_tracked_ast = layers_selec[layers_selec==asthenosphere_code].size
n_tracked_mantle_lithosphere1 = layers_selec[layers_selec==mantle_lithosphere1_code].size
n_tracked_mantle_lithosphere2 = layers_selec[layers_selec==mantle_lithosphere2_code].size
n_tracked_lower_crust = layers_selec[layers_selec==lower_crust_code].size
n_tracked_upper_crust = layers_selec[layers_selec==upper_crust_code].size

print(f"Asthenosphere: {n_tracked_ast}")
print(f"Mantle Lithosphere: {n_tracked_mantle_lithosphere1+n_tracked_mantle_lithosphere2}")
print(f"Upper Crust: {n_tracked_upper_crust}")
print(f"Lower Crust: {n_tracked_lower_crust}\n")

if(n_tracked == 0):
    raise ValueError("No particles selected. Check search_thickness, x_begin, x_end and instant_to_take.")

#Reading the whole history of the selected particles at once
start = 0
end = idx
print(f'Start idx: {start}, End idx: {end}, time of end: {Tdataset.time.values[end]}')

nc_idx = [steps_nc_index(s) for s in Tdataset.steps.values[start:end+1]]
#Reading the whole (time, particle) block and indexing with numpy is much faster than .sel(particle=...) on the file
part_idx = np.searchsorted(Sdataset.particle.values, part_selec)
xtrack = Sdataset.x.values[nc_idx][:, part_idx] #(time, particle)
ztrack = Sdataset.z.values[nc_idx][:, part_idx]

n_missing = np.count_nonzero(~np.isfinite(xtrack))
if(n_missing > 0):
    print(f"WARNING: {n_missing} (time, particle) positions do not exist in _steps.nc. P and T will be NaN there.")

#We need shared pymp arrays to store data.
ptrack = pymp.shared.array(xtrack.shape, dtype=np.float64)
ttrack = pymp.shared.array(xtrack.shape, dtype=np.float64)
ptrack[:] = np.nan
ttrack[:] = np.nan

#Interpolating P and T at the particle positions
with pymp.Parallel() as p:
    for i in p.range(start, end+1):
        it = i - start
        temperature = Tdataset.temperature[i].values
        pressure = Pdataset.pressure[i].values

        vecx_track = xtrack[it]
        vecz_track = ztrack[it]
        ok = np.isfinite(vecx_track)

        jm = np.clip(((vecx_track[ok]-0)//dx).astype(int), 0, Nx-2)
        im = np.clip(((vecz_track[ok]+Lz)//dz).astype(int), 0, Nz-2)

        xw = (vecx_track[ok] - xx[im,jm])/dx
        zw = (vecz_track[ok] - zz[im,jm])/dz

        present_temperature =   temperature[im,jm]*    (1.-xw)*(1.-zw)+\
                                temperature[im,jm+1]*  (xw   )*(1.-zw)+\
                                temperature[im+1,jm]*  (1.-xw)*(   zw)+\
                                temperature[im+1,jm+1]*(   xw)*(   zw)

        present_pressure =  pressure[im,jm]*    (1.-xw)*(1.-zw)+\
                            pressure[im,jm+1]*  (xw   )*(1.-zw)+\
                            pressure[im+1,jm]*  (1.-xw)*(   zw)+\
                            pressure[im+1,jm+1]*(   xw)*(   zw)

        ptrack[it, ok] = present_pressure / 1.0e6
        ttrack[it, ok] = present_temperature

        np.savetxt(f"track_xzPT_step_{str(Tdataset.steps.values[i]).zfill(6)}.txt", np.c_[vecx_track, vecz_track, ptrack[it], ttrack[it]], fmt="%.5f")

all_time = Tdataset.time.values[start:end+1]
all_step = Tdataset.steps.values[start:end+1]

#Flattening (time, particle) as in track_particles.py: time ascending, then reversed
all_vecx_track = xtrack.ravel()
all_vecz_track = ztrack.ravel()
all_present_pressure = ptrack.ravel()
all_present_temperature = ttrack.ravel()

print(f"len of:\n all_vecx_track {len(all_vecx_track)}\n all_vecz_track {len(all_vecz_track)}\n all_present_pressure {len(all_present_pressure)}\n all_present_temperature {len(all_present_temperature)}\n all_time {len(all_time)}\n all_step {len(all_step)}")
# Creatiing the xarray dataset with the tracked particles
ds = xr.Dataset(
    {
        "xtrack": (["index"], all_vecx_track[::-1]),
        "ztrack": (["index"], all_vecz_track[::-1]),
        "ptrack": (["index"], all_present_pressure[::-1]),
        "ttrack": (["index"], all_present_temperature[::-1]),
        "step": all_step[::-1],
        "time": all_time[::-1],
        "ntracked": int(n_tracked),
        "particles_layers": layers_selec[::-1]
    },
    coords={
        "index": np.arange(len(all_vecx_track))
    }
)

ds.to_netcdf("_track_xzPT_all_steps.nc")


print(f"n_tracked x len(all_time) = {n_tracked}*{len(all_time)} = {n_tracked*len(all_time)}")

# Compressing the files
print("Zipping files")
subprocess.run(f"zip {model_name}.zip track*.txt", shell=True, check=True, capture_output=True, text=True)
subprocess.run(f"rm track*.txt", shell=True, check=True, capture_output=True, text=True)
print("Files zipped")
