#basic libs
import os, gc, json, sys, glob
from pathlib import Path
# sys.path.append('/media/jobueno/STOV/scripts/')
#from matplotlib.collections import LineCollection

#matplotlib and mpl objects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


from scipy.interpolate import RegularGridInterpolator


import numpy as np
import pandas as pd
import xarray as xr


#from easyRo_class import easyRo

#cdir = "/media/jobueno/STOV/new_salt_sys/" #os.getcwd()
#cmap_dir = '/media/jobueno/STOV/scripts/salt_cmap.json'
#os.chdir(cdir)

#PROCESSOR - GEMINI STRUCTURE

#Vars
_varsTypes = {'density': np.float32, #smaller float / float
              'pressure': np.float32,
              'heat': np.float32,
              'thermal_difussivity': np.float32,
              'surface': np.float32,
              'lithology': np.int8, #128 layers / signed char
              'viscosity': np.float64, #bigger float / double
              'velocity': np.float64,
              'strain': np.float64,
              'strain_rate':np.float64,
              #Dimensions
              'time': np.float32,
              'x': np.int32,
              'z': np.int32}

class MandyocProcessor:
    def __init__(self, model_path):
        """
        Inicializa o processador lendo dinamicamente os parâmetros estruturais
        do modelo para evitar 'hard code'.
        """
        self.model_path = Path(model_path)
        self.params = self._read_params(self.model_path / "param.txt")
        
        # Define as dimensões da grade com base no param.txt
        self.nx = int(self.params.get('nx', 0))
        self.nz = int(self.params.get('nz', 0))
        self.lx = float(self.params.get('lx', 0.0))
        self.lz = float(self.params.get('lz', 0.0))
        
        # Coordenadas espaciais regulares
        self.x_coords = np.linspace(0, self.lx, self.nx)
        self.z_coords = np.linspace(-self.lz, 0, self.nz) # Assumindo z negativo para profundidade
        
        # Mapeamento de tempo e passos
        self.steps, self.times = self._get_time_steps()
        self.num_steps = len(self.steps)
        
        # Verifica quais variáveis existem organizadas em pastas pelo mv-updated.sh
        self.available_vars = [d.name for d in self.model_path.iterdir() if d.is_dir()]

    def _read_params(self, param_file):
        """Lê o param.txt e retorna um dicionário com os parâmetros."""
        params = {}
        with open(param_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                line = line.split('#')[0].replace(' ', '')
                if '=' in line:
                    k, v = line.split('=', 1)
                    params[k.lower()] = v
        return params

    def _get_time_steps(self):
        """Descobre todos os passos de tempo disponíveis lendo a pasta 'time'."""
        time_dir = self.model_path / "time"
        files = glob.glob(str(time_dir / "time_*.txt"))
        steps = sorted([int(Path(f).stem.split('_')[1]) for f in files])
        
        times = []
        for step in steps:
            with open(time_dir / f"time_{step}.txt", 'r') as f:
                # Extrai o tempo em Myr
                val = float(f.readline().split()[1]) / 1e6 
                times.append(val)
        return steps, times

    def process_nodal_variable(self, var_name, n_cores=4):
        """
        Processa propriedades contínuas (density, temperature, etc.) usando pymp
        para paralelizar a leitura dos arquivos de texto.
        """
        if var_name not in self.available_vars:
            print(f"Variável {var_name} não encontrada.")
            return

        print(f"Processando {var_name} em paralelo com {n_cores} threads...")
        
        # Aloca um array compartilhado na memória para todas as threads escreverem
        # Dimensões: (tempo, z, x) - Padrão geofísico de profundidade no eixo y do array
        shared_data = pymp.shared.array((self.num_steps, self.nz, self.nx), dtype='float32')
        
        var_dir = self.model_path / var_name
        
        with pymp.Parallel(n_cores) as p:
            for i in p.range(self.num_steps):
                step = self.steps[i]
                file_path = var_dir / f"{var_name}_{step}.txt"
                
                try:
                    # Carrega pulando o cabeçalho 'P'
                    data_step = np.loadtxt(file_path, comments="P", skiprows=3, dtype='float32')
                    # Zera valores ínfimos
                    data_step[np.abs(data_step) < 1e-200] = 0.0
                    
                    # O Mandyoc geralmente exporta ordenado em x, depois z. Precisamos fazer reshape e transpor.
                    reshaped_data = data_step.reshape((self.nx, self.nz)).T
                    shared_data[i, :, :] = reshaped_data
                except FileNotFoundError:
                    # Preenche com NaN caso algum passo esteja corrompido/ausente
                    shared_data[i, :, :] = np.nan

        self._save_netcdf(var_name, np.array(shared_data), is_2d=True)

    def process_surface(self):
        """Processa a topografia, que é um array 1D no espaço."""
        if "surface" not in self.available_vars:
            return
            
        surface_data = np.zeros((self.num_steps, self.nx), dtype='float32')
        surf_dir = self.model_path / "surface"
        
        for i, step in enumerate(self.steps):
            file_path = surf_dir / f"surface_{step}.txt"
            # O arquivo de superfície pode não ter skiprows=3
            data = np.loadtxt(file_path, comments="P", dtype='float32')
            # Extração da topografia (sy)
            if data.ndim == 2:
                surface_data[i, :] = data[:, 1] # Pega a coluna Y
            else:
                surface_data[i, :] = data
                
        self._save_netcdf("surface", surface_data, is_2d=False)

    def process_particles(self, prefix="step", n_cores=4):
        """
        Processa nuvens de partículas, agrupando os arquivos distribuídos por núcleos 
        do Mandyoc (ex: step_0_0.txt, step_0_1.txt).
        """
        print(f"Processando partículas com prefixo '{prefix}'...")
        
        # Descobre número de cores usados na simulação original olhando para o passo 0
        part_dir = self.model_path / ("lithos" if prefix == "litho" else prefix)
        n_model_cores = len(list(part_dir.glob(f"{prefix}_0_*.txt")))
        
        if n_model_cores == 0:
            return
            
        # Estrutura para salvar DataFrames processados
        all_particles = []
        
        for i, step in enumerate(self.steps):
            dfs = []
            for core in range(n_model_cores):
                fpath = part_dir / f"{prefix}_{step}_{core}.txt"
                if fpath.exists():
                    df = pd.read_csv(fpath, sep='\s+', comment='P', header=None, 
                                     names=['x', 'z', 'id', 'layer', 'strain'])
                    df['time'] = self.times[i]
                    dfs.append(df)
            
            if dfs:
                step_df = pd.concat(dfs, ignore_index=True)
                all_particles.append(step_df)
                
        if all_particles:
            particles_total = pd.concat(all_particles)
            # Indexação hierárquica por ID e Tempo
            particles_total.set_index(['id', 'time'], inplace=True)
            ds = particles_total.to_xarray()
            
            # Compressão zlib
            comp = dict(zlib=True, complevel=4)
            encoding = {var: comp for var in ds.data_vars}
            
            out_file = self.model_path / f"_output_particles_{prefix}.nc"
            ds.to_netcdf(out_file, encoding=encoding)
            print(f"Partículas salvas em {out_file}")

    def _save_netcdf(self, var_name, data_array, is_2d=True):
        """
        Constrói o xarray.Dataset e salva em disco com compressão.
        """
        dims = ["time", "z", "x"] if is_2d else ["time", "x"]
        coords = {"time": self.times, "x": self.x_coords}
        if is_2d:
            coords["z"] = self.z_coords

        ds = xr.Dataset(
            {var_name: (dims, data_array)},
            coords=coords,
            attrs={"description": f"Processed {var_name} from Mandyoc",
                   "lx": self.lx, "lz": self.lz}
        )
        
        # Aplica compressão no NetCDF4
        encoding = {var_name: {'zlib': True, 'complevel': 4, '_FillValue': np.nan}}
        
        out_file = self.model_path / f"_output_{var_name}.nc"
        ds.to_netcdf(out_file, encoding=encoding, engine='netcdf4')
        print(f"Salvo: {out_file}")

'''
# ====== Exemplo de Uso ======
if __name__ == "__main__":
    caminho_modelo = "./meu_cenario_mandyoc"
    
    # 1. Instancia o objeto (lê a geometria automaticamente)
    processor = MandyocProcessor(caminho_modelo)
    
    # 2. Processa as matrizes contínuas (paralelizado no tempo)
    variaveis_nodais = ['density', 'temperature', 'viscosity', 'strain_rate', 'pressure']
    for var in variaveis_nodais:
        processor.process_nodal_variable(var, n_cores=8)
        
    # 3. Processa topografia (1D temporal)
    processor.process_surface()
    
    # 4. Processa partículas (se for simulação termomecânica com marcadores)
    processor.process_particles(prefix="step", n_cores=4)
'''

#Mandyoc Scenario class - Need improvements
class MandyocScen:
    
    def __init__(self, path, variables = ['density'], cmap_dir='/media/jobueno/STOV/scripts/salt_cmap.json', 
                 xlimits=None,ylimits=None, thick_air=40e3, domain_ranges={}, 
                 chunks_vars={"x":'auto',"z":'auto','time':"auto"},
                 verbose=False):
        
        self.path = path
        #self.read_cmap(cmap_dir)
        
        if not isinstance(variables, (list, tuple, np.ndarray)): variables = [variables]
        if len(variables) > 0: self.get_scenarioData(variables[0])
        else: print('No variable was given, it can cause bugs')
        
        self.xlimits = xlimits if xlimits is not None else [self.XMIN, self.XMAX]
        self.ylimits = ylimits if ylimits is not None else [self.ZMIN, self.ZMAX]
        
        self.thick_air = thick_air #m
        self.z_corrected = False
        self.particles_load = False
        self.verbose = verbose #future implementations -> colocar isso nas funções
        
        self.vars_DS = {} #dictionary with the loaded variables
        self.domains = domain_ranges #dict -> future implementations using particles over time
        
        self.original_particles = None   #whole particles dataset (can be replaced for subsets)
        self.selected_particles = None   #current selected particles
        self.particles = {}  #dictionary with particle selections
        
        for var in variables:
            self.load_var(var, chunks=chunks_vars)
        
        #self.read_cmap(cmap_dir)
        self.cmap = None
        self.cmap_metadata = None
        
        return None
    
    def get_scenarioData(self, var='density'):
        path = self.path
        DS = xr.open_dataset(f'{os.path.join(path,f"_{var}")}.nc')[f'{var}']
        self.Nx = int(DS.x.count())
        self.Nz = int(DS.z.count())
        self.XMAX = int(DS.x.max())
        self.XMIN = int(DS.x.min())
        self.ZMAX = int(DS.z.max())
        self.ZMIN = int(DS.z.min())
        self.TMAX = int(DS.time.max())
        self.TMIN = int(DS.time.min())
        DS.close()
        return True
    
    def read_cmap(self, cmap_dir):
        json_dir = cmap_dir
        file = open(json_dir,"r")
        cmap_json = json.load(file)
        self.cmap_metadata = cmap_json['metadata']
        self.cmap = LinearSegmentedColormap.from_list(name=self.cmap_metadata['name'], 
                                                      colors=cmap_json['colors'], 
                                                      N=cmap_json['metadata']['N_layers'])
        #plt.register_cmap(name=cmap_metadata['name'], cmap=cmap_json['colors'])
        return True

    def correctZcoord(self, factor=None):
        if self.z_corrected:
            print("Z was already corrected")
            return False
        
        if factor == None: #factor is the value to subtract from Z axis
            factor = int(self.ZMAX - self.thick_air)
            #print(factor)
        
        for v, ds in self.vars_DS.items():
            self.vars_DS[v]["z"] = ("z", (ds['z'].data - factor))
        
        if self.particles_load == True:
            self.original_particles["z"] = self.original_particles["z"] - factor
        self.z_corrected = True
        #self.selected_particles["z"] = ("z", (self.selected_particles['z'].data - factor))
        
        return True

    def load_var(self, variable, chunks={}):
        path = os.path.join(self.path,f'_{variable}')
        v = xr.open_dataset(f"{path}.nc", chunks=chunks)
        v = v.sel(x=slice(self.xlimits[0],self.xlimits[-1]),
              z=slice(self.ylimits[0],self.ylimits[-1]))
        self.vars_DS[variable] = v
        return True
    
    def load_mainParticles(self, name='particles_trajectories.nc', chunks={'id':'auto'}, filter_air=True):
        path = os.path.join(self.path,name)
        self.original_particles = xr.open_dataset(path, chunks=chunks)
        if filter_air==True:
            air = int(self.original_particles.layer.max())
            cond = self.original_particles.layer != air
            self.original_particles = self.original_particles.where(cond)
        self.particles_load = True
        return self
    
    def _apply_selection(self, ids_validos, select_original, replace_original, name_selection):
        """
        Método interno para aplicar uma lista de IDs filtrados ao dataset de trajetórias.
        Centraliza a lógica de atribuição e limpeza de memória.
        """
        # 1. Escolha da fonte de trajetórias completas (ID, TIME)
        if select_original:
            source_ds = self.original_particles
        else:
            source_ds = self.selected_particles
    
        # 2. Aplica a seleção de IDs
        # Usamos .sel(id=...) para manter a trajetória completa dos IDs filtrados
        filtered_ds = source_ds.sel(id=ids_validos)
    
        # 3. Atribuição de resultados conforme as flags
        if replace_original:
            self.original_particles = filtered_ds
        else:
            self.selected_particles = filtered_ds
    
        if len(name_selection) > 0:
            self.particles[name_selection] = filtered_ds
    
        # Limpeza de memória mandatória
        gc.collect()
        return ids_validos

    
    #Selecting particles
    def selectParticles_bytimerange(self, timerange, select_original=True, replace_original=False, 
                                particles=None, name_selection=''):
        '''
        select particles that appeared within the specified time ranges
        timerange : list = [tmin, tmax]
        '''
        
        if select_original==True: pts = self.original_particles
        else: pts = self.particles[particles]
        
        #pts_bk = pts.copy()
        tr_0 = pts.sel(time=timerange[0],method='nearest').dropna(dim="id").id.values
        tr_i = pts.sel(time=timerange[1],method='nearest').dropna(dim="id").id.values
        
        ids = list(set(tr_i)-set(tr_0))
        
        #print('ok')
        self._apply_selection(ids, select_original, replace_original, name_selection)
        return self
        
    
    def selectParticles_bycoords(self, xlim=None, ylim=None, tsel=None, select_original=True, 
                                 replace_original=False, particles=None, name_selection=''):
        '''
        coords : list = [[xmin, xmax],[ymin,ymax]]
        '''
        
        if xlim is None: xlim = self.xlimits
        if ylim is None: ylim = self.ylimits
        
        pts = self.original_particles if select_original else self.selected_particles
            
        if tsel is None: tsel = float(pts.attrs['reference timestep'].split(' ')[0][:-3]) #myr
        
        #pts_bk = pts.copy()
        pts = pts.sel(time=tsel, method='nearest')
        condX = ((pts["x"] >= xlim[0]) & (pts["x"] <= xlim[1])).compute()
        pts =  pts.where(condX, drop=True)
        
        condZ = ((pts["z"] >= ylim[0]) & (pts["z"] <= ylim[1])).compute()
        pts =  pts.where(condZ, drop=True)
        ids = pts.id.values
              
        self._apply_selection(ids, select_original, replace_original, name_selection)
        return self
    
    
    def selectParticles_bylayers(self, layers, tsel=None, select_original=True, 
                                 replace_original=False, particles=None, name_selection=''):
        '''
        select layers by particles
        '''
        
        pts = self.original_particles if select_original else self.selected_particles
        if tsel is None: tsel = float(pts.attrs['reference timestep'].split(' ')[0][:-3]) #myr
        
        #pts_bk = pts.copy()
        pts = pts.sel(time=tsel, method='nearest')
        cond = pts.layer.isin(layers).compute()
        ids =  pts.id.where(cond, drop=True).values
        
        self._apply_selection(ids, select_original, replace_original, name_selection)
        return self
    
    
    def classify_ParticlesRange(self, domain_intervals, tsel=None, select_original=True, 
                                 replace_original=False, particles=None, name_selection=''):
        
        #Classify all particles based on X ranges, given a time step tsel
        #Categories are based on the domain intervals keys
        
        pts = self.original_particles if select_original else self.selected_particles
        if tsel is None: tsel = float(pts.attrs['reference timestep'].split(' ')[0][:-3]) #myr
        
        domain_intervals = domain_intervals.copy()
        
        try: field_name = domain_intervals['field_name']
        except: field_name = 'domain'
        del domain_intervals['field_name']
        
        typename = type(list(domain_intervals.keys())[1])
        if typename is str: typename='U256'
        
        if field_name in pts:
            print(f"{field_name} is a dataset variable, choose another name")
            return False
        
        pts[field_name] = (['id'], np.full(pts.sizes['id'], '', dtype=typename))
        
        snapshot = pts.sel(time=tsel, method='nearest')
        
        for dom, intervals in list(domain_intervals.items()):
            
            if not isinstance(intervals[0], (list, tuple, np.ndarray)): intervals = [intervals]
            mask_combined = np.zeros(snapshot.sizes['id'], dtype=bool)
            
            for start, end in intervals:
                mask_current = ((snapshot.x >= start) & (snapshot.x <= end)).compute()
                mask_combined |= mask_current.values
            
            ids_in_domain = snapshot.id.values[mask_combined]
            pts[field_name] = xr.where(pts.id.isin(ids_in_domain), dom,  pts[field_name])
        
        
        if replace_original:
            self.original_particles = pts
        else:
            self.selected_particles = pts
            
        if len(name_selection) > 0:
            self.particles[name_selection] = pts
        
        pts[field_name].attrs['reference timestep'] = f'{tsel}myr'
        pts[field_name].attrs['classes range'] = str(domain_intervals)
        gc.collect()
        
        return self
    
    def _field_interpolate_outdated(self,field_ds,time_i,components,buffer=10e3,verbose=False,particles='temp'):
        time_i=int(time_i)
        if particles=='temp':
            particles = self._tempParticles
        pts_i = particles.isel(time=time_i)
        var_i = field_ds.isel(time=time_i)
        
        if verbose is True:
            print(f"step: {time_i}")
        
        px=pts_i.x.values
        pz=pts_i.z.values
        
        minX, maxX = np.nanmin(px)-buffer,np.nanmax(px)+buffer
        minZ, maxZ = np.nanmin(pz)-buffer,np.nanmax(pz)+buffer
        
        gridXZ= var_i.sel(x=slice(minX, maxX),z=slice(minZ, maxZ))
        grid_x = gridXZ['x'].values.astype('int32')
        grid_z = gridXZ['z'].values.astype('int32')
        pts_target = np.column_stack((pz, px))
        
        for comp in components:
            gridInterp = gridXZ[comp].values
            interpObj = RegularGridInterpolator((grid_z, grid_x), gridInterp, 
                                                 bounds_error=False, fill_value=np.nan)
            particles[comp][:, time_i] = interpObj(pts_target)
            
        return True
    
    def fieldToParticle_outdated(self,variable, buffer=10e3, verbose=False, select_original=True, 
                                 replace_original=False, particles=None, name_selection=''):
        
        pts0 = self.original_particles if select_original else self.selected_particles
        field = self.vars_DS[variable]
        #if not isinstance(variables, (list, tuple, np.ndarray)): variables = [variable]
        
        pts = pts0.copy()
        times = pts.time.values
        
        components = list(field.data_vars)
        for comp in components:
            if comp not in pts:
                pts[comp] = (['id', 'time'], np.full((pts.sizes['id'], pts.sizes['time']), 
                                                     np.nan, dtype=field[comp].dtype))
                
        self._tempParticles = pts
        if verbose is True:
            print(f"processing {variable}")
            print(f"total steps: {int(times.size)}")
        
        for t_i in range(len(times)):
           self._field_interpolate(field,t_i,components, verbose=verbose,particles='temp')
           #interpolated_values.append(pts_var)
        
        if replace_original is True:
            self.original_particles = self._tempParticles
        else:
            self.selected_particles = self._tempParticles
            
        if len(name_selection) > 0:
            self.particles[name_selection] = self._tempParticles
        
        self._tempParticles = None
        gc.collect()
        return self

    def fieldToParticle(self, variable, select_original=True, 
                                 replace_original=False, particles=None, name_selection=''):
        
        pts0 = self.original_particles if select_original else self.selected_particles
        field = self.vars_DS[variable]
        #if not isinstance(variables, (list, tuple, np.ndarray)): variables = [variable]
        
        pts = pts0.copy()
        
        components = list(field.data_vars)
        
        field_interpolated = field.interp(
            x=pts.x, 
            z=pts.z,
            time=pts.time,
            method='linear'
        )
        
        for comp in components:
            #print('comp')
            #print(field_interpolated[comp])
            pts[comp] = field_interpolated[comp].drop_vars(['x', 'z']).transpose('id', 'time') #did it worked?
            # pts[comp] = (['id', 'time'], field_interpolated[comp].data)
        
        
        if replace_original is True:
            self.original_particles = pts
        else:
            self.selected_particles = pts
            
        if len(name_selection) > 0:
            self.particles[name_selection] = pts
        
        gc.collect()
        return self

class mandyocPlotter(): #função para receber os dados de um cenario e fazer plots
    
    def __init__(self, mandyocScen):
        
        return self
    
    
    class snapshot(): #sub classe de Plotter para plotar os campos/cenarios de 1 step selecionado
        
        def __init__(self, tselect):
        
            
            return self
        


'''
side_thresh = 750e3 #m
domains = {'field_name' : 'domain',
                   'proximal' : ([300e3,430e3],[1075e3,1200e3]),  #15m2
                  'transition' : ([430e3,510e3],[1030e3,1075e3]),
                  'distal': ([510e3,side_thresh],[side_thresh,1030e3])}


kpsalt_15m2 = MandyocScen("/media/jobueno/STOV/new_salt_sys/kp_salt/15m2/", 
                           xlimits=[200e3,1200e3], ylimits=[220e3,270e3],
                           variables=['temperature','strain_rate','viscosity','lithology'])

kpsalt_15m2.load_mainParticles(name="particles_trajectories.nc",chunks={'id':50000},filter_air=True)
kpsalt_15m2.correctZcoord()
kpsalt_15m2.ylimits = [-25e3, 5e3]
sides = {"field_name":'side',
        "left":[kpsalt_15m2.XMIN,side_thresh],
        "right": [side_thresh,kpsalt_15m2.XMAX]}

kpsalt_15m2.selectParticles_bycoords(replace_original=True)
kpsalt_15m2.selectParticles_bylayers([5,6,7, 11,8,9,10],tsel=69.4, replace_original=True)

kpsalt_15m2.fieldToParticle('temperature',verbose=True,select_original=True,replace_original=True)

kpsalt_15m2.classify_ParticlesRange(domain_intervals=sides, 
                                    tsel=69.5, replace_original=True)
kpsalt_15m2.classify_ParticlesRange(domain_intervals=domains, tsel=69.5, replace_original=True)

#export to a new nc
#kpsalt_15m2.original_particles.to_netcdf(f"{kpsalt_15m2.path}/sed_particles_classified.nc")

kpsalt_15m2.selectParticles_bycoords(ylim=[-0.5e3, 1e3],
                                     tsel=0,
                                     name_selection='pre-rift')
gc.collect()

kpsalt_15m2.selectParticles_bytimerange([20,25], name_selection='time 20-25')
kpsalt_15m2.selectParticles_bytimerange([40,45], name_selection='time 40-45')

#bug
#kpsalt_15m2.particles['time 40-45'] = kpsalt_15m2.particles['time 40-45'].where(kpsalt_15m2.particles['time 40-45'].layer!=3)

kpsalt_15m2.selectParticles_bylayers([8, 9, 10],tsel=17, name_selection='syn-rift')
kpsalt_15m2.selected_particles.layer.compute().max()

kpsalt_15m2.particles['pre-salt'] = xr.concat([kpsalt_15m2.particles['syn-rift'],
                                               kpsalt_15m2.particles['pre-rift']],
                                               dim='id')

#kpsalt_15m2.selected_particles = kpsalt_15m2.particles['sediments'].copy()


#Função do GEMINI
dominios = ['proximal', 'transition', 'distal']
lados = ['left', 'right']

# Dicionário mapeando o nome da sequência para a sua cor e para o dataset correspondente
# (Substitua as chamadas do kpsalt_15m2 pelos seus objetos reais)
seqs_info = {
    'pre-salt':   {'cor': 'black',  'ds': kpsalt_15m2.particles['pre-salt']},
    'time 20-25': {'cor': 'orange', 'ds': kpsalt_15m2.particles['time 20-25']},
    'time 40-45': {'cor': 'brown',   'ds': kpsalt_15m2.particles['time 40-45']}
}

# 2. Configuração da Figura
fig, axs = plt.subplots(len(dominios), len(lados), figsize=(12, 10), 
                        sharex=True, sharey=True)

# 3. Loops de Plotagem
for i, dominio in enumerate(dominios):  # Itera pelas linhas
    for j, lado in enumerate(lados):    # Itera pelas colunas
        ax = axs[i, j]
        
        # Itera sobre cada sequência (para colocar as 3 cores na mesma caixa)
        for seq_name, info in seqs_info.items():
            ds = info['ds']
            cor = info['cor']
            
            # Cria a máscara filtrando lado e domínio simultaneamente
            # O .compute() é obrigatório aqui pois suas variáveis são dask.arrays
            mask = (ds['side'] == lado) & (ds['domain'] == dominio)
            mask = mask.compute()
            
            # Filtra os IDs (muito mais eficiente do que o ds.where(..., drop=True) genérico)
            ids_validos = ds.id.values[mask]
            
            if len(ids_validos) > 0:
                ds_filtrado = ds.sel(id=ids_validos)
                
                # A. Plota as trajetórias de fundo (Spaghetti) com matriz transposta (.T)
                ax.plot(ds_filtrado.time.values, 
                        ds_filtrado['temperature'].values.T, 
                        color=cor, alpha=0.05)
                
                # B. Plota a mediana em destaque (linha grossa)
                mediana = ds_filtrado['temperature'].median(dim='id')
                ax.plot(ds_filtrado.time.values, mediana, 
                        color=cor, linewidth=2.5, label=seq_name)

        # 4. Estética dos Eixos
        # Títulos nas colunas (Margens)
        if i == 0:
            titulo = "left margin" if lado == 'left' else "right margin"
            ax.set_title(titulo, fontsize=14, fontweight='bold')
            
        # Nomes nas linhas (Domínios)
        if j == 0:
            ax.set_ylabel(f'{dominio.capitalize()}\nTemp [°C]', fontsize=12, fontweight='bold')
            
        # Eixo X apenas no fundo
        if i == len(dominios) - 1:
            ax.set_xlabel('Time [Myr]', fontsize=12)

# 5. Legenda Global Unificada
# Extrai as legendas de um dos eixos para não repetir
handles, labels = axs[0, 0].get_legend_handles_labels()
# Usa dict para remover as labels duplicadas (caso a mediana tenha sido plotada várias vezes)
by_label = dict(zip(labels, handles))

# Coloca a legenda no topo da figura, fora dos gráficos
fig.legend(by_label.values(), by_label.keys(), loc='upper center', 
           bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12, 
           title="Sediment sequence")

plt.tight_layout()
plt.show()
'''
