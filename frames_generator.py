#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import multiprocessing #needed to run pymp in mac
import psutil
multiprocessing.set_start_method('fork') #needed to run pymp in mac 
# multiprocessing.set_start_method('spawn')
import pymp
import subprocess
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


matplotlib.use('agg')

path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path

path_to_functions = f"{machine_path}/opt/rifting_melt"
sys.path.append(os.path.abspath(path_to_functions))

if '' in sys.path:
    sys.path.remove('')
from functions.mandyocIO import read_datasets, change_dataset, plot_property

####################################################################################################################################
model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")


if not os.path.isdir(output_path):
    os.makedirs(output_path)

plot_isotherms = True
# plot_isotherms = False
plot_melt = True
# plot_melt = False
plot_particles=False


if("4" in model_name and "0" in model_name):
    hcrust = 40.0e3 #m
else:
    hcrust = 35.0e3 #m

if(plot_isotherms or plot_melt):
    clean_plot=False
else:
    clean_plot = True

datasets = [#Properties from mandyoc. Comment/uncomment to select properties of the dataset
            # 'density',
            # 'radiogenic_heat',
            # 'pressure',
            'strain',
            # 'strain_rate',### Read ascii outputs and save them as xarray.Datasets,
            # 'surface',
            'temperature',
            # 'viscosity'
            ]# Read data and convert them to xarray.Dataset

properties = [#Properties from mandyoc. Comment/uncomment to select which ones you would like to plot
            #  'density',
            #  'radiogenic_heat',
             'lithology',
            #  'pressure',
            #  'strain',
            #  'strain_rate',
            #  'temperature',
            #  'temperature_anomaly',
            #  'surface',
            #  'viscosity'
             ]

#######################################################
# Read ascii outputs and save them as xarray.Datasets #
#######################################################

new_datasets = change_dataset(properties, datasets)
# print(new_datasets)
to_remove = []
remove_density=False
if ('density' not in properties): #used to plot air/curst interface
        properties.append('density')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('density')
        # remove_density=True

# if ('surface' not in properties): #used to plot air/curst interface
#         properties.append('surface')
#         new_datasets = change_dataset(properties, datasets)
#         to_remove.append('surface')
        # remove_density=True

if (plot_isotherms): #add datasets needed to plot isotherms
    if ('temperature' not in new_datasets):
        properties.append('temperature')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('temperature')

# print(f"newdatasets: {new_datasets}")

if (plot_melt): #add datasets needed to plot melt fraction
    if ('melt' not in properties):
        properties.append('melt')
    if ('incremental_melt' not in properties):
        properties.append('incremental_melt')
    new_datasets = change_dataset(properties, datasets)

    #removing the auxiliary datasets to not plot
    to_remove.append('melt')
    to_remove.append('incremental_melt')

if(clean_plot): #a clean plot
    new_datasets = change_dataset(properties, datasets)

for item in to_remove:
    properties.remove(item)
    
# dataset = read_datasets(model_path, new_datasets)
# print(dataset.keys())
# Normalize velocity values
# if ("velocity_x" and "velocity_z") in dataset.data_vars:
#     v_max = np.max((dataset.velocity_x**2 + dataset.velocity_z**2)**(0.5))    
#     dataset.velocity_x[:] = dataset.velocity_x[:] / v_max
#     dataset.velocity_z[:] = dataset.velocity_z[:] / v_max

# if ('lithology' in properties):
# lithology_dataset = xr.open_dataset(f"{model_path}/_lithology.nc")
# strain_dataset = xr.open_dataset(f"{model_path}/_strain.nc")
# temperature_dataset = xr.open_dataset(f"{model_path}/_temperature.nc")

# if(plot_melt):
#     Phi_dataset = xr.open_dataset(f"{model_path}/_melt.nc")
#     dPhi_dataset = xr.open_dataset(f"{model_path}/_incremental_melt.nc")

#########################################
# Get domain and particles informations #
#########################################

Nx = int(xr.open_dataset(f"{model_path}/_strain.nc").nx) #int(strain_dataset.nx)
Nz = int(xr.open_dataset(f"{model_path}/_strain.nc").nz) #int(strain_dataset.nz)
Lx = float(xr.open_dataset(f"{model_path}/_strain.nc").lx) #float(strain_dataset.lx)
Lz = float(xr.open_dataset(f"{model_path}/_strain.nc").lz) #float(strain_dataset.lz)

x = np.linspace(0, Lx/1000.0, Nx)
z = np.linspace(-Lz/1000.0, 0, Nz)
xx, zz  = np.meshgrid(x, z)

# print(particles_layers)
############################################################################################################################
# Plotting
plot_colorbar = True
h_air = 40.0

start = 0
end = int(xr.open_dataset(f"{model_path}/_strain.nc").time.size)
step = 1

make_videos = True
# make_videos = False

make_gifs = True
# make_gifs = False

zip_files = True
# zip_files = False

print("Generating frames...")

color_lower_crust='xkcd:brown'

color_incremental_melt = 'xkcd:bright pink'
color_depleted_mantle='xkcd:bright purple'
# topo_from_density = False
topo_from_density = True

linewidth = 0.1
markersize = 4
line_alpha = 1.0
# color_crust='xkcd:grey'

# color_incremental_melt = 'xkcd:bright pink'
# color_depleted_mantle='xkcd:dark grey'

cr = 255.
color_air = (1.,1.,1.) # 5
color_bas = (250./cr,50./cr,50./cr) # 4
color_uc = (228./cr,156./cr,124./cr) # 3
color_lc = (240./cr,209./cr,188./cr) # 2
color_lit = (155./cr,194./cr,155./cr) # 1
color_ast = (207./cr,226./cr,205./cr) # 0


colors = [color_ast,
          color_lit,
          color_lc,
          color_uc,
        #   color_bas,
          color_air]

#Creating a custom colormap according to the list of colors defined above.
# This colormap will be used to plot the lithology mesh, where each lithology type is represented by a specific color.

cmap = ListedColormap(colors)
np.seterr(divide='ignore')
# time = time[::-1]
steps = np.array(xr.open_dataset(f"_strain.nc").steps.values)
times = np.array(xr.open_dataset(f"_strain.nc").time.values)
print(f'nx, nz, times: {Nx}, {Nz}, {len(times)}, type')
fig_format = 'jpeg'

def worker(i):
    lithology = xr.open_dataset("_lithology.nc", cache=False, engine="h5netcdf")["lithology"].isel(time=i).values[::-1,:]
    strain  = xr.open_dataset("_strain.nc", cache=False, engine="h5netcdf")["strain"].isel(time=i).values[::-1,:]
    temperature = xr.open_dataset("_temperature.nc", cache=False, engine="h5netcdf")["temperature"].isel(time=i).values
    return lithology, strain, temperature

A = np.zeros((100, 10))

A[:25, :] = 2700
A[25:50, :] = 2800
A[50:75, :] = 3300
A[75:100, :] = 3400

A = A[::-1, :]

xA = np.linspace(-0.5, 0.9, 10)
yA = np.linspace(0, 1.5, 100)

xxA, yyA = np.meshgrid(xA, yA)

chunk_size = 4

with pymp.Parallel() as p:
    # for i in p.range(start, end-step, step):
    for chunk_start in p.range(start, end, chunk_size):
        lithology = xr.open_dataset("_lithology.nc", cache=False, engine="h5netcdf")["lithology"]
        strain = xr.open_dataset("_strain.nc", cache=False, engine="h5netcdf")["strain"]
        temperature = xr.open_dataset("_temperature.nc", cache=False, engine="h5netcdf")["temperature"]
        if(plot_melt):
            incremental_melt = xr.open_dataset("_incremental_melt.nc", cache=False, engine="h5netcdf")["dPhi"]
            melt = xr.open_dataset("_melt.nc", cache=False, engine="h5netcdf")["Phi"]

        for i in range(chunk_start, min(chunk_start + chunk_size, end)):
            steps_model = steps[i] #dataset.steps.values[i]
            fig, axs = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)

            # lithology, strain, temperature = worker(i)

            current_time = float(times[i])
            
            xlims = [0, Lx/1000]
            ylims = [-Lz/1000+40, 0+40]
            axs.text(0.01, 1.035, f'{model_name}', bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=axs.transAxes)
            axs.text(0.5, 1.035, f'Time = {current_time:.2f} Myr', bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=axs.transAxes)
            
            axs.imshow(lithology.isel(time=i).values[::-1, :], aspect='auto', extent=(0, Lx/1000, -Lz/1000+40, 40), cmap=cmap, vmin=0, vmax=5, alpha=1.0)  

            # strain  = strain_dataset.strain.values[i][::-1,:]
            # temperature = temperature_dataset.temperature.values[i][::-1,:]   

            # axs.imshow(np.log10(strain), extent=(0, Lx/1000, -Lz/1000+40, 40), cmap="Greys", vmin=-0.5, vmax=0.9, alpha=0.2)
            axs.imshow(np.log10(strain.isel(time=i).values[::-1, :]), extent=(0, Lx/1000, -Lz/1000+40, 40), cmap="Greys", vmin=-0.5, vmax=0.9, alpha=0.2)
            axs.contour(x, z+40, temperature.isel(time=i).values, levels=[500, 600, 700, 800, 900, 1300], colors='r', linewidths=1.0)
            axs.set_ylim(ylims)
            axs.set_xlim(xlims)

            bbox_to_anchor=(0.90,#horizontal position respective to parent_bbox or "loc" position
                            0.20,# vertical position
                            0.08,# width
                            0.25)
        
            # bv1 = inset_axes(axs,
            #                 loc='lower right',
            #                 width="100%",  # respective to parent_bbox width
            #                 height="100%",  # respective to parent_bbox width
            #                 bbox_to_anchor=bbox_to_anchor,
            #                 bbox_transform=axs.transAxes
            #                 )
            bv1 = fig.add_axes([0.9,#horizontal position respective to parent_bbox or "loc" position
                            0.40,# vertical position
                            0.07,# width
                            0.15])

            air_threshold = 200
            bv1.contourf(
                xxA,
                yyA,
                A,
                levels=[air_threshold, 2750, 2900, 3365, 3900],
                colors=[color_uc, color_lc, color_lit, color_ast],
                extent=[-0.5, 0.9, 0, 1.5]
            )

            bv1.imshow(
                xxA[::-1, :],
                extent=[-0.5, 0.9, 0, 1.5],
                zorder=100,
                alpha=0.2,
                cmap=plt.get_cmap("Greys"),
                vmin=-0.5,
                vmax=0.9,
                aspect='auto'
            )

            bv1.set_yticklabels([])
            bv1.set_xlabel(r"log$(\varepsilon_{II})$", size=10)
            bv1.tick_params(axis='x', which='major', labelsize=10)
            bv1.set_xticks([-0.5, 0, 0.5])
            bv1.set_yticks([])
            bv1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # Setting plot details
            fsize = 14
            axs.set_xlabel('Distance [km]', fontsize=fsize)
            axs.set_ylabel('Depth [km]', fontsize=fsize)
            axs.tick_params(axis='both', labelsize=fsize)

            if(plot_melt):
                # incremental_melt = xr.open_dataset('_incremental_melt.nc', cache=False, engine="h5netcdf")['dPhi'].isel(time=i).values[::,:] #dPhi_dataset.isel(time=i).to_numpy()[::,:]
                # melt = xr.open_dataset('_melt.nc', cache=False, engine="h5netcdf")['Phi'].isel(time=i).values[::,:] #Phi_dataset.isel(time=i).to_numpy()[::,:]
                incremental_melt_i = incremental_melt.isel(time=i).values[::,:]
                melt_i = melt.isel(time=i).values[::,:]
                #Plotting incremental melt
                color_incremental_melt = 'xkcd:bright pink'
                color_depleted_mantle='xkcd:purple'

                meltmin, meltmax = melt_i.min(), melt_i.max()
                dmeltmin, dmeltmax = incremental_melt_i.min(), incremental_melt_i.max()
                incremental_melt_i[incremental_melt_i == 0] = np.nan # Set zero values to NaN to avoid plotting them
                melt_i[melt_i == 0] = np.nan # Set zero values to NaN to avoid plotting them

                axs.contourf(xx, zz+40, incremental_melt_i, colors=color_incremental_melt, alpha=0.4, zorder=30)
                axs.contourf(xx, zz+40, melt_i, colors=color_depleted_mantle, alpha=0.4, zorder=20)
                #plotting melt legend
                text_fsize = 12
                axs.text(0.10, 1.035, r'Melt Fraction $\left(\frac{\partial \phi}{\partial t}\right)$', color='xkcd:bright pink', fontsize=text_fsize, transform=axs.transAxes, zorder=60)
                axs.text(0.25, 1.035, r'Depleted Mantle ($\phi$)', color='xkcd:bright purple', fontsize=text_fsize, transform=axs.transAxes, zorder=60)

                figname = f"{model_name}_lithology_and_PTt_MeltFrac_{str(int(steps_model)).zfill(6)}.{fig_format}"
            else:
                figname = f"{model_name}_lithology_and_PTt_{str(int(steps_model)).zfill(6)}.{fig_format}"

            proc = psutil.Process(os.getpid())
            print(f"before - worker ID:{os.getpid()}; Memory Usage: {proc.memory_info().rss/1024**3:.2f} GB")
            fig.savefig(f"_output/{figname}", dpi=300)
            print(f"after - worker ID:{os.getpid()}; Memory Usage: {proc.memory_info().rss/1024**3:.2f} GB")
            # print('saved')
            # print(f'callbacks: {fig.canvas.callbacks.callbacks}')
            plt.close(fig)

            del fig
            del axs
            del bv1
            gc.collect()

        del lithology
        del strain
        del temperature
        if(plot_melt):
            del incremental_melt
            del melt
        gc.collect()

        # objs = gc.get_objects()
        # print(f'figures remaining: {sum(isinstance(o, Figure) for o in objs)}')

print("Done!")

##############################################################################################################################################################################
if(make_videos):
    print("Generating videos...")

    fps = 24
    for prop in properties:
        videoname = f'{model_path}/_output/{model_name}_lithology_and_PTt'

        if(plot_melt):
            videoname = f'{videoname}_MeltFrac'

        if(plot_particles):
            if(prop == 'viscosity'):
                videoname = f'{videoname}'
            else:
                videoname = f'{videoname}_particles'
                # videoname = f'{videoname}_particles_onlymb'
            
        try:
            comand = f"rm {videoname}.mp4"
            result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
            print(f"\tRemoving previous {prop} video.")
        except:
            print(f"\tNo {prop} video to remove.")

        comand = f"ffmpeg -r {fps} -f image2 -s 1920x1080 -pattern_type glob -i \"{videoname}_*.{fig_format}\" -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an -crf 25 -pix_fmt yuv420p {videoname}.mp4"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
    print("\tDone!")


##########################################################################################################################################################################

# # Converting videos to gifs
# 
# ss: skip seconds
# 
# t: duration time of the output
# 
# i: inputs format
# 
# vf: filtergraph (video filters)
# 
#     - fps: frames per second
# 
#     - scale: resize accordint to given pixels (e.g. 1080 = 1080p wide)
#     
#     - lanczos: scaling algorithm
#     
#     - palettegen and palette use: filters that generate a custom palette
#     
#     - split: filter that allows everything to be done in one command
# 
# loop: number of loops
# 
#     - 0: infinite
# 
#     - -1: no looping
# 
#     - for numbers n >= 0, create n+1 loops


if(make_gifs):
    print("Converting videos to gifs...")
    for prop in properties:
        gifname = f'{model_path}/_output/{model_name}_lithology_and_PTt'

        if(plot_melt):
            gifname = f'{gifname}_MeltFrac'

        if(plot_particles):
            if(prop == 'viscosity'):
                gifname = f'{gifname}'
            else:
                gifname = f'{gifname}_particles'
                # gifname = f'{gifname}_particles_onlymb'
            

        try:
            comand = f"rm {gifname}.gif"
            result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
            print(f"\tRemoving previous {prop} gif.")
        except:
            print(f"\tNo {prop} gif to remove.")
        
        comand = f"ffmpeg -ss 0 -t 15 -i '{gifname}.mp4' -vf \"fps=30,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 {gifname}.gif"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True) 
    print("\tDone!")

##########################################################################################################################################################################

if(zip_files):
    #zip plots, videos and gifs
    print('Zipping figures, videos and gifs...')
    outputs_path = f'{model_path}/_output/'
    os.chdir(outputs_path)
    subprocess.run(f"zip {model_name}_imgs.zip *.{fig_format}", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_videos.zip *.mp4", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_gifs.zip *.gif", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"rm *.{fig_format}", shell=True, check=True, capture_output=True, text=True)
    print('Zipping complete!')
    os.chdir(f'{model_path}')