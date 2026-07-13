#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import multiprocessing #needed to run pymp in mac
from multiprocessing import Pool
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
strain_ds = None
temperature_ds = None
lithology_ds = None
melt_ds = None
dphi_ds = None

def init_worker(model_path, plot_melt):

    global strain_ds, temperature_ds, lithology_ds
    global melt_ds, dphi_ds

    strain_ds = xr.open_dataset(
        f"{model_path}/_strain.nc",
        cache=False
    )

    temperature_ds = xr.open_dataset(
        f"{model_path}/_temperature.nc",
        cache=False
    )

    lithology_ds = xr.open_dataset(
        f"{model_path}/_lithology.nc",
        cache=False
    )
    if plot_melt:
            melt_ds = xr.open_dataset(
                f"{model_path}/_melt.nc",
                cache=False
            )

            dphi_ds = xr.open_dataset(
                f"{model_path}/_incremental_melt.nc",
                cache=False
            )

# def plot_frame(i):

def plot_chunks(indexs):

    global strain_ds
    global temperature_ds
    global lithology_ds
    global melt_ds
    global dphi_ds

    np.seterr(divide='ignore')
    
    Nx = int(strain_ds.nx) #int(strain_dataset.nx)
    Nz = int(strain_ds.nz) #int(strain_dataset.nz)
    Lx = float(strain_ds.lx) #float(strain_dataset.lx)
    Lz = float(strain_ds.lz) #float(strain_dataset.lz)

    x = np.linspace(0, Lx/1000.0, Nx)
    z = np.linspace(-Lz/1000.0, 0, Nz)
    xx, zz  = np.meshgrid(x, z)

    steps = np.array(strain_ds.steps.values)
    times = np.array(strain_ds.time.values)

    linewidth = 0.1
    markersize = 4
    line_alpha = 1.0

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
    
    rhos = np.zeros((100, 10))

    rhos[:25, :] = 2700
    rhos[25:50, :] = 2800
    rhos[50:75, :] = 3300
    rhos[75:100, :] = 3400

    rhos = rhos[::-1, :]

    xA = np.linspace(-0.5, 0.9, 10)
    yA = np.linspace(0, 1.5, 100)

    xxA, yyA = np.meshgrid(xA, yA)

    #Creating a custom colormap according to the list of colors defined above.
    # This colormap will be used to plot the lithology mesh, where each lithology type is represented by a specific color.

    cmap = ListedColormap(colors)
    for i in indexs:
        lithology = lithology_ds["lithology"].isel(time=i).values[::-1, :]
        strain = strain_ds["strain"].isel(time=i).values[::-1, :]
        temperature = temperature_ds["temperature"].isel(time=i).values

        fig, axs = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)

        current_time = float(times[i])
        steps_model = float(steps[i])

        xlims = [0, Lx/1000]
        ylims = [-Lz/1000+40, 0+40]
        axs.text(0.01, 1.035, f'{model_name}', bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=axs.transAxes)
        axs.text(0.5, 1.035, f'Time = {current_time:.2f} Myr', bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=axs.transAxes)
        
        axs.imshow(lithology, aspect='auto', extent=(0, Lx/1000, -Lz/1000+40, 40), cmap=cmap, vmin=0, vmax=5, alpha=1.0)  

        axs.imshow(np.log10(strain), extent=(0, Lx/1000, -Lz/1000+40, 40), cmap="Greys", vmin=-0.5, vmax=0.9, alpha=0.2)
        axs.contour(x, z+40, temperature, levels=[500, 600, 700, 800, 900, 1300], colors='r', linewidths=1.0)
        axs.set_ylim(ylims)
        axs.set_xlim(xlims)

        bv1 = fig.add_axes([0.9,#horizontal position respective to parent_bbox or "loc" position
                        0.40,# vertical position
                        0.07,# width
                        0.15])

        air_threshold = 200

        bv1.contourf(
            xxA,
            yyA,
            rhos,
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
            incremental_melt = xr.open_dataset('_incremental_melt.nc', cache=False, engine="h5netcdf")['dPhi'].isel(time=i).values[::,:] #dPhi_dataset.isel(time=i).to_numpy()[::,:]
            melt = xr.open_dataset('_melt.nc', cache=False, engine="h5netcdf")['Phi'].isel(time=i).values[::,:] #Phi_dataset.isel(time=i).to_numpy()[::,:]

            #Plotting incremental melt
            color_incremental_melt = 'xkcd:bright pink'
            color_depleted_mantle='xkcd:purple'

            meltmin, meltmax = melt.min(), melt.max()
            dmeltmin, dmeltmax = incremental_melt.min(), incremental_melt.max()
            incremental_melt[incremental_melt == 0] = np.nan # Set zero values to NaN to avoid plotting them
            melt[melt == 0] = np.nan # Set zero values to NaN to avoid plotting them

            axs.contourf(xx, zz+40, incremental_melt, colors=color_incremental_melt, alpha=0.4, zorder=30)
            axs.contourf(xx, zz+40, melt, colors=color_depleted_mantle, alpha=0.4, zorder=20)
            #plotting melt legend
            text_fsize = 12
            axs.text(0.01, 1.035, r'Melt Fraction $\left(\frac{\partial \phi}{\partial t}\right)$', color='xkcd:bright pink', fontsize=text_fsize, transform=axs.transAxes, zorder=60)
            axs.text(0.21, 1.035, r'Depleted Mantle ($\phi$)', color='xkcd:bright purple', fontsize=text_fsize, transform=axs.transAxes, zorder=60)

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
        del fig, axs, bv1, lithology, strain, temperature
        if(plot_melt):
            del incremental_melt, melt
        gc.collect()

plot_colorbar = True
h_air = 40.0

start = 0
end = int(xr.open_dataset(f"{model_path}/_strain.nc").time.size)
step = 1

frames = range(start, end-step)
fig_format = 'jpeg'

chunks = np.array_split(
    np.arange(start, end-step),
    12
)

with Pool(
    processes=12,
    initializer=init_worker, #initializer function to call when starting a new process
    initargs=(model_path, plot_melt), #arguments to pass to the initializer function
    maxtasksperchild=10
) as pool:
    pool.map(plot_chunks, chunks)

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