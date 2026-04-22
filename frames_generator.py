#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import glob
import multiprocessing #needed to run pymp in mac
multiprocessing.set_start_method('fork') #needed to run pymp in mac
import pymp
import subprocess
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
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
from functions.mandyocIO import read_datasets, change_dataset, plot_property, find_nearest

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

if(plot_isotherms or plot_melt):
    clean_plot=False
else:
    clean_plot = True

datasets = [#Properties from mandyoc. Comment/uncomment to select properties of the dataset
            'density',
            'radiogenic_heat',
            'pressure',
            'strain',
            'strain_rate',### Read ascii outputs and save them as xarray.Datasets,
            # 'surface',
            'temperature',
            'viscosity'
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

if ('surface' not in properties): #used to plot air/curst interface
        properties.append('surface')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('surface')
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
    
dataset = read_datasets(model_path, new_datasets)

# Normalize velocity values
if ("velocity_x" and "velocity_z") in dataset.data_vars:
    v_max = np.max((dataset.velocity_x**2 + dataset.velocity_z**2)**(0.5))    
    dataset.velocity_x[:] = dataset.velocity_x[:] / v_max
    dataset.velocity_z[:] = dataset.velocity_z[:] / v_max

lithology_dataset = xr.open_dataset(f"{model_path}/_lithology.nc")
#########################################
# Get domain and particles informations #
#########################################

Nx = int(len(dataset.x))
Nz = int(len(dataset.z))
Lx = float(dataset.x[-1]) #km
Lz = float(dataset.z[-1]) #km

x = np.linspace(0, Lx/1000.0, Nx)
z = np.linspace(Lz/1000.0, 0, Nz)
xx, zz  = np.meshgrid(x, z)

# double_keel = True
double_keel = False
sediments = False
# sediments = True

print(f"double_keel model") if(double_keel) else print("homogeneous model")

if(double_keel):
    if(sediments==True):
        asthenosphere_code = 0
        lower_craton_code = 1
        upper_craton_code = 2
        mantle_lithosphere_code = 3
        lower_crust1_code = 4
        seed_code = 5
        lower_crust2_code = 6
        upper_crust_code = 7
        decolement_code = 8
        sediments_code = 9
        air_code = 10
    else:
        asthenosphere_code = 0
        lower_craton_code = 1
        upper_craton_code = 2
        mantle_lithosphere_code = 3
        lower_crust1_code = 4
        seed_code = 5
        lower_crust2_code = 6
        upper_crust_code = 7
        air_code = 8
else:
    asthenosphere_code = 0
    mantle_lithosphere_code = 1
    lower_crust1_code = 2
    seed_code = 3
    lower_crust2_code = 4
    upper_crust_code = 5
    decolement_code = 6
    sediments_code = 7
    air_code = 8

############################################################################################################################
# Plotting
plot_colorbar = True
h_air = 40.0

t0 = dataset.time[0]
t1 = dataset.time[1]
dt = int(t1 - t0)

start = int(t0)
end = int(dataset.time.size - 1)
step = 2

# ts = glob.glob(f"{model_path}/time/time_*.txt")
# steps = []
# for t in ts:
#  step = t.split('/')[-1].split('.')[0].split('_')[-1]
#  steps.append(int(step))

steps = dataset.steps.values

# start = 4
# end = 5
# step = 1

make_videos = True
# make_videos = False

make_gifs = True
# make_gifs = False

zip_files = True
# zip_files = False

print("Generating frames...")
linewidth = 0.1
markersize = 4
line_alpha = 1.0
# color_crust='xkcd:grey'

color_incremental_melt = 'xkcd:bright pink'
color_depleted_mantle='xkcd:dark grey'
cr = 255.
color_air = (1.,1.,1.) # 5
color_bas = (250./cr,50./cr,50./cr) # 4
color_uc = (228./cr,156./cr,124./cr) # 3
color_lc = (240./cr,209./cr,188./cr) # 2
color_lit = (155./cr,194./cr,155./cr) # 1
color_ast = (207./cr,226./cr,205./cr) # 0


colors = [color_ast,color_lit,color_lc,color_uc,color_bas,color_air]

#Creating a custom colormap according to the list of colors defined above.
# This colormap will be used to plot the lithology mesh, where each lithology type is represented by a specific color.

cmap = ListedColormap(colors)

with pymp.Parallel() as p:
    for i in p.range(start, end+step, step):
        # instant = np.round(dataset.time.values[i], 2)
        instant = dataset.time.isel(time=i).values
        # data = dataset.isel(time=i)
        # current_time = float(data.time.values)
        # print(f"Plotting frame {i} of {end} at time {current_time} Myr...")
        for prop in properties:
            fig, axs = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
            axs.text(0.5, 1.035, f'Time = {instant:.2f} Myr', bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=axs.transAxes)
            # xlims = [0, float(data.lx) / 1.0e3]
            # ylims = [-float(data.lz) / 1.0e3 + 40, 40s]
            # ylims = [-150, 40]
            # ylims = [-400, 40]

            data = lithology_dataset['lithology'].isel(time=i).to_numpy()[::-1,:]
            axs.imshow(data, extent=[0, Lx/1000, Lz/1000, 0], cmap=cmap, vmin=0, vmax=5, alpha=1.0)
            axs.imshow(np.log10(dataset.strain.isel(time=i)[::-1,:]), extent=[0,Lx/1000,Lz/1000,0], cmap="Greys", vmin=-0.5, vmax=0.9, alpha=0.2)
            axs.contour(dataset.temperature.isel(time=i)[::-1,:], levels=[500, 600, 700,800, 900, 1300], colors='r', linewidths=1.0)
            

            melt = dataset.Phi.isel(time=i).to_numpy()[::,:]
            incremental_melt = dataset.dPhi.isel(time=i).to_numpy()[::,:]
            meltmin, meltmax = melt.min(), melt.max()
            dmeltmin, dmeltmax = incremental_melt.min(), incremental_melt.max()

            incremental_melt[incremental_melt == 0] = np.nan # Set zero values to NaN to avoid plotting them
            melt[melt == 0] = np.nan # Set zero values to NaN to avoid plotting them

            axs.contourf(xx, zz, incremental_melt, levels=0, colors=color_incremental_melt, alpha=0.4, zorder=30)
            axs.contour(xx, zz, incremental_melt, levels=1, colors=color_incremental_melt, linewidths=1.5, alpha=1.0, zorder=30)

            axs.contourf(xx, zz, melt, levels=0, colors=color_depleted_mantle, alpha=0.4, zorder=20)
            axs.contour(xx, zz, melt, levels=2, colors='xkcd:blue', linewidths=0.8, alpha=0.8, zorder=20)

            bbox_to_anchor=(0.93,#horizontal position respective to parent_bbox or "loc" position
                0.25,# vertical position
                0.065,# width
                0.30)
            
            bv1 = inset_axes(axs,
                            loc='lower right',
                            width="100%",  # respective to parent_bbox width
                            height="100%",  # respective to parent_bbox width
                            bbox_to_anchor=bbox_to_anchor,
                            bbox_transform=axs.transAxes
                            )
            
            A = np.zeros((100, 10))

            A[:25, :] = 2700
            A[25:50, :] = 2800
            A[50:75, :] = 3300
            A[75:100, :] = 3400

            A = A[::-1, :]

            xA = np.linspace(-0.5, 0.9, 10)
            yA = np.linspace(0, 1.5, 100)

            xxA, yyA = np.meshgrid(xA, yA)
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

            fsize = 14
            axs.set_xlabel('Distance [km]', fontsize=fsize)
            axs.set_ylabel('Depth [km]', fontsize=fsize)
            axs.tick_params(axis='both', labelsize=fsize)

            if(plot_melt):
                #plotting melt legend
                text_fsize = 12
                axs.text(0.01, 0.90, r'Melt Fraction $\left(\frac{\partial \phi}{\partial t}\right)$', color='xkcd:bright pink', fontsize=text_fsize, transform=axs.transAxes, zorder=60)
                axs.text(0.15, 0.90, r'Depleted Mantle ($\phi$)', color='xkcd:bright purple', fontsize=text_fsize, transform=axs.transAxes, zorder=60)

                figname = f"{model_name}_{prop}_and_PTt_temperature_coded_MeltFrac_{str(int(steps[i])).zfill(6)}.png"
            else:
                figname = f"{model_name}_{prop}_and_PTt_temperature_coded_{str(int(steps[i])).zfill(6)}.png"
            fig.savefig(f"_output/{figname}", dpi=300)
            plt.close('all')

        del data
        gc.collect()

print("Done!")

##############################################################################################################################################################################
if(make_videos):
    print("Generating videos...")

    fps = 24
    for prop in properties:
        videoname = f'{model_path}/_output/{model_name}_{prop}_and_PTt_temperature_coded'

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

        comand = f"ffmpeg -r {fps} -f image2 -s 1920x1080 -pattern_type glob -i \"{videoname}_*.png\" -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an -crf 25 -pix_fmt yuv420p {videoname}.mp4"
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
        gifname = f'{model_path}/_output/{model_name}_{prop}_and_PTt_temperature_coded'

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
    subprocess.run(f"zip {model_name}_imgs.zip *.png", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_videos.zip *.mp4", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_gifs.zip *.gif", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"rm *.png", shell=True, check=True, capture_output=True, text=True)
    print('Zipping complete!')