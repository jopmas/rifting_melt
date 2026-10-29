"""
Readers for the NetCDF files created from Mandyoc outputs.

They open, with the same structure, the files created by:

- tapioca (``meshes2netcdf.jl`` and ``particles2netcdf.jl``): ``strain.nc``, ``Phi.nc``, ``lithology.nc``,
  ``particles_trajectories.nc``, ...
- the old Julia scripts (``convertNETCDF_v2.jl``, ``LithoNETCDF_v2.jl`` and ``StepsNETCDF.jl``): ``_strain.nc``,
  ``_melt.nc``, ``_lithology.nc``, ``_steps.nc``, ...
- the old python scripts (``mandyocIO.read_mandyoc_output``): ``_output_strain.nc``, ...

Every mesh dataset is returned with tapioca's structure:

- data with dims (time, z, x), so ``ds.strain.isel(time=i).values`` is (Nz, Nx) and the first row is the bottom;
- ``z`` from -Lz to 0 and ``x`` from 0 to Lx [m];
- the step number in the coordinate ``step`` (the old files call it ``steps``);
- vertical velocity in ``vz`` (the old files call it ``vy``);
- the attributes ``nx``, ``nz``, ``lx`` and ``lz``.

Values are not changed when opening. Use ``mask_air`` and ``legacy_lithology_codes`` to reproduce the processing
done by the old Julia scripts (they set to zero all values in the air and remapped the lithology codes).

Particles are returned by ``open_particles`` with the structure of ``_steps.nc``: dims (time, particle), ``x`` and
``z`` with NaN where the particle does not exist, ``layer`` (particle) and the coordinate ``step``.
"""

import os
import numpy as np
import xarray as xr

# Names accepted in open_mandyoc and the name of the variable inside the file
ALIASES = {
    "melt": "Phi",
    "incremental_melt": "dPhi",
    "depletion_factor": "X_depletion",
    "radiogenic_heat": "heat",
}

# Old file names when they are not "_{var}.nc"
LEGACY_FILES = {
    "Phi": ["_melt.nc", "_output_melt.nc"],
    "dPhi": ["_incremental_melt.nc", "_output_incremental_melt.nc"],
    "X_depletion": ["_depletion_factor.nc", "_output_depletion_factor.nc"],
    "heat": ["_heat.nc", "_output_radiogenic_heat.nc"],
}

# The old Julia scripts set to zero every value where density < AIR_DENSITY
AIR_DENSITY = 1200.0

# Lithology codes remapping done by LithoNETCDF_v2.jl (Mandyoc code -> old code). Codes not listed are kept.
LEGACY_LITHOLOGY_MAP = {
    2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
}

_LEGACY_LITHOLOGY_LUT = np.arange(256, dtype=np.int16) - 128 # index = code + 128
for _new, _old in LEGACY_LITHOLOGY_MAP.items():
    _LEGACY_LITHOLOGY_LUT[_new + 128] = _old


def _canonical(var):
    return ALIASES.get(var, var)


def find_file(model_path, var, prefer="tapioca"):
    '''
    Find the NetCDF file of a Mandyoc variable.

    Parameters
    ----------
    model_path: str
        Folder of the scenario.
    var: str
        Variable name (e.g. 'strain', 'Phi' or 'melt', 'lithology', 'velocity').
    prefer: str
        'tapioca' to use the tapioca file when both exist, or 'legacy' to use the old one.

    Returns
    -------
    path: str
        Path of the file.
    '''
    var = _canonical(var)
    tapioca = [f"{var}.nc"]
    legacy = LEGACY_FILES.get(var, [f"_{var}.nc", f"_output_{var}.nc"])
    candidates = tapioca + legacy if prefer == "tapioca" else legacy + tapioca

    for fname in candidates:
        path = os.path.join(model_path, fname)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(f"No NetCDF file for '{var}' in {model_path}. Tried: {', '.join(candidates)}")


def file_format(path):
    '''
    'tapioca' for files from tapioca, 'legacy' for files from convertNETCDF_v2.jl/LithoNETCDF_v2.jl,
    'legacy_python' for files from mandyocIO (_output_*.nc).
    '''
    fname = os.path.basename(path)
    if fname.startswith("_output_"):
        return "legacy_python"
    if fname.startswith("_"):
        return "legacy"
    return "tapioca"


def normalize_mesh(ds, fmt):
    '''
    Change a mesh dataset to tapioca's structure (see the module docstring). It does not load the data.
    '''
    if "steps" in ds.variables and "step" not in ds.variables:
        ds = ds.rename({"steps": "step"})
    if "step" in ds.data_vars:
        ds = ds.set_coords("step")
    if "vy" in ds.data_vars and "vz" not in ds.data_vars:
        ds = ds.rename({"vy": "vz"})

    # (time, x, z) -> (time, z, x)
    for v in ds.data_vars:
        if set(ds[v].dims) == {"time", "x", "z"}:
            ds[v] = ds[v].transpose("time", "z", "x")

    if "z" in ds.coords and ds.sizes.get("z", 0) > 1:
        z = ds.z.values
        if z[0] >= 0 and z[-1] > 0: # old Julia files: z from 0 to Lz
            ds = ds.assign_coords(z=z - z[-1])

    if "x" in ds.coords:
        ds.attrs["nx"] = int(ds.sizes["x"])
        ds.attrs["lx"] = float(ds.x.values[-1] - ds.x.values[0])
    if "z" in ds.coords:
        ds.attrs["nz"] = int(ds.sizes["z"])
        ds.attrs["lz"] = float(ds.z.values[-1] - ds.z.values[0])

    ds.attrs["mandyoc_format"] = fmt
    for v in ds.data_vars:
        ds[v].attrs["mandyoc_format"] = fmt

    return ds


def open_mandyoc(model_path, var, prefer="tapioca", **kwargs):
    '''
    Open the NetCDF file of a Mandyoc variable with tapioca's structure, whatever the script that created it.

    Parameters
    ----------
    model_path: str
        Folder of the scenario.
    var: str
        Variable name. Old names are accepted: 'melt' (Phi), 'incremental_melt' (dPhi),
        'depletion_factor' (X_depletion) and 'radiogenic_heat' (heat).
    prefer: str
        'tapioca' or 'legacy', used when both files exist.
    kwargs:
        Passed to xarray.open_dataset (e.g. cache=False, engine="h5netcdf", chunks={"time": 1}).

    Returns
    -------
    ds: xarray.Dataset
    '''
    path = find_file(model_path, var, prefer=prefer)
    ds = normalize_mesh(xr.open_dataset(path, **kwargs), file_format(path))

    # _lithology.nc from LithoNETCDF_v2.jl has no steps: take them from times.csv (written by the Julia scripts)
    times_csv = os.path.join(model_path, "times.csv")
    if "step" not in ds.coords and "time" in ds.dims and os.path.isfile(times_csv):
        steps, times = np.loadtxt(times_csv, delimiter=",", skiprows=1, unpack=True, ndmin=2)
        if steps.size == ds.sizes["time"] and np.allclose(times, ds.time.values, atol=1e-3):
            ds = ds.assign_coords(step=("time", steps.astype(np.int32)))

    return ds


def mesh_info(model_path, prefer="tapioca"):
    '''
    Returns Nx, Nz, Lx, Lz of the scenario reading the first file found among density, temperature and strain.
    '''
    for var in ["density", "temperature", "strain", "pressure", "viscosity"]:
        try:
            with open_mandyoc(model_path, var, prefer=prefer) as ds:
                return int(ds.nx), int(ds.nz), float(ds.lx), float(ds.lz)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No mesh NetCDF file found in {model_path}")


def mask_air(data, density, value=0.0):
    '''
    Set `value` where density < AIR_DENSITY, as done by convertNETCDF_v2.jl.

    Data from the old Julia scripts already have zeros in the air, so they are returned unchanged.

    Parameters
    ----------
    data: xarray.DataArray
        Data from open_mandyoc (e.g. ds.strain.isel(time=i)). It can have all time steps: open both files
        with chunks={"time": 1} to keep it lazy.
    density: xarray.DataArray or numpy.ndarray
        Density in the same mesh and time of `data`.
    '''
    if data.attrs.get("mandyoc_format") == "legacy":
        return data
    if not isinstance(density, xr.DataArray):
        density = np.asarray(density)
    return data.where(~(density < AIR_DENSITY), value)


def legacy_lithology_codes(lithology):
    '''
    Remap Mandyoc lithology codes as done by LithoNETCDF_v2.jl (see LEGACY_LITHOLOGY_MAP), so colormaps made
    for the old _lithology.nc files keep working.

    Files from LithoNETCDF_v2.jl are already remapped, so they are returned unchanged.

    Parameters
    ----------
    lithology: xarray.DataArray
        Lithology from open_mandyoc (e.g. ds.lithology.isel(time=i)).
    '''
    if lithology.attrs.get("mandyoc_format") == "legacy":
        return lithology
    codes = np.asarray(lithology.values)
    remapped = _LEGACY_LITHOLOGY_LUT[codes.astype(np.int16) + 128].astype(codes.dtype)
    return lithology.copy(data=remapped)


def open_particles(model_path, prefer="tapioca", **kwargs):
    '''
    Open the particles NetCDF file with the structure of _steps.nc.

    Reads particles_trajectories.nc (tapioca's particles2netcdf.jl) or _steps.nc (StepsNETCDF.jl).

    Returns
    -------
    ds: xarray.Dataset
        x, z (time, particle) [m] with NaN where the particle does not exist; layer (particle) with the Mandyoc
        lithology code; coordinates particle (id), time [Myr] and step.
    '''
    candidates = ["particles_trajectories.nc", "_steps.nc"]
    if prefer != "tapioca":
        candidates = candidates[::-1]
    for fname in candidates:
        path = os.path.join(model_path, fname)
        if os.path.isfile(path):
            break
    else:
        raise FileNotFoundError(f"No particles NetCDF file in {model_path}. Tried: {', '.join(candidates)}")

    ds = xr.open_dataset(path, **kwargs)

    if fname == "_steps.nc":
        ds = ds.rename({"steps": "step"}).set_coords("step")
        ds.attrs["mandyoc_format"] = "legacy"
        return ds.transpose("time", "particle")

    # Files created before fixing the dimension of "id" (it was "ID")
    if "ID" in ds.dims:
        ids = ds["id"].values
        ds = ds.drop_vars("id").assign_coords(id=("id", ids))

    ds = ds.reset_coords([v for v in ["x", "z"] if v in ds.coords])
    if "step" in ds.data_vars:
        ds = ds.set_coords("step")

    exists = ds.x != -999.0
    lithology = ds.lithology.where(ds.lithology >= 0).max("time")

    out = xr.Dataset(
        {
            "x": ds.x.where(exists),
            "z": ds.z.where(exists),
            "layer": lithology.fillna(-1).astype(ds.lithology.dtype),
        },
        coords={"step": ds.step},
        attrs=dict(ds.attrs, mandyoc_format="tapioca"),
    )
    out = out.rename({"id": "particle"}).transpose("time", "particle")
    out["layer"].attrs["long_name"] = "Layer code"
    return out
