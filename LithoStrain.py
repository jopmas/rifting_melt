import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import sys
import glob

cr = 255.
color_air = (1.,1.,1.) # 5
color_bas = (250./cr,50./cr,50./cr) # 4
color_uc = (228./cr,156./cr,124./cr) # 3
color_lc = (240./cr,209./cr,188./cr) # 2
color_lit = (155./cr,194./cr,155./cr) # 1
color_ast = (207./cr,226./cr,205./cr) # 0


colors = [color_ast,color_lit,color_lc,color_uc,color_bas,color_air]

cmap = ListedColormap(colors)

def replace_negatives_with_neighbors(mat):
    # Copy to avoid modifying during iteration
    result = mat.copy()
    rows, cols = mat.shape

    for i in range(rows):
        for j in range(cols):
            if result[i, j] < 0:
                # Check up, down, left, right for non-negative neighbor
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and mat[ni, nj] >= 0:
                        result[i, j] = mat[ni, nj]
                        break
    return result



step_initial = int(sys.argv[1])
step_final = int(sys.argv[2])

d_step = int(sys.argv[3])
ncores = int(sys.argv[4])


with open("param.txt","r") as f:
	line = f.readline()
	line = line.split()
	Nx = int(line[2])
	line = f.readline()
	line = line.split()
	Nz = int(line[2])
	line = f.readline()
	line = line.split()
	Lx = float(line[2])
	line = f.readline()
	line = line.split()
	Lz = float(line[2])

print(Nx,Nz,Lx,Lz) #Lz/(Nz-1)


Nxl = (Nx-1)*5
Nzl = (Nz-1)*5



#xx,zz = np.mgrid[0:Lx:(Nx)*1j,-Lz:0:(Nz)*1j]

xi = np.linspace(0,Lx/1000,Nx)
zi = np.linspace(-Lz/1000,0,Nz)
xx,zz = np.meshgrid(xi,zi)


ts = glob.glob("Tempo_*.txt")
total_curves = len(ts) #total_steps/print_step
n_curves = total_curves/2

val = 100





for cont in range(step_initial,step_final,d_step):#
	print(cont)

	litho_mesh = np.zeros((Nxl,Nzl))-1

	A = np.loadtxt("time_"+str(cont)+".txt",dtype='str')  
	AA = A[:,2:]
	AAA = AA.astype("float") 
	tempo = np.copy(AAA)
	
	print("Time = %.1lf Myr\n\n"%(tempo[0]/1.0E6))

	
	A = pd.read_csv("density_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = TT[:,:]
	TTT = np.transpose(TTT)
	rho = np.copy(TTT)
    

	A = pd.read_csv("temperature_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = np.transpose(TT)
	temper = np.copy(TTT)


	A = pd.read_csv("strain_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = np.transpose(TT)
	TTT[rho<200]=0
	#TTT[rho>3365]=0
	#TTT[TTT<1.0E-1]=0
	TTT = np.log10(TTT)
	stc = np.copy(TTT)
	
	x = []
	z = []
	litho = []
	for core in range(ncores):
		A = pd.read_csv("litho_%d_%d.txt"%(cont,core),delimiter = " ",comment="P",skiprows=0,header=None) 
		A = A.to_numpy()
		print(np.shape(A))
		x = np.append(x,A[:,0])
		z = np.append(z,A[:,1])
		litho = np.append(litho,A[:,2])

	z = z.astype(int)
	x = x.astype(int)

    
	litho_mesh[x,z] = litho

	litho_mesh = replace_negatives_with_neighbors(litho_mesh)
     
	
	
	litho_mesh[litho_mesh==2]=1 #Lithospheric mantle
	litho_mesh[litho_mesh==3]=1 #Lithospheric mantle
	litho_mesh[litho_mesh==4]=2 #Lower crust
	litho_mesh[litho_mesh==5]=3 #Upper crust
	litho_mesh[litho_mesh==6]=4 #Bas
	litho_mesh[litho_mesh==7]=5 #Air
	
	print("basalto",np.sum(litho_mesh==4))

	plt.close()
	plt.figure(figsize=(10*2,2.5*2))

	

	
	plt.imshow(np.transpose(litho_mesh),extent=[0,Lx/1000,-Lz/1000,0],cmap=cmap,vmin=0,vmax=5,interpolation="none")
	
	#para plotar as isotermas em vermelho
	plt.contour(xx,zz,temper,levels=[550,750,850,950,1200,1300],
		colors=[(1,0,0),(1,0,0),(1,0,0)],linewidths=1.0)

	print("stc",np.min(stc),np.max(stc))

	#stc = np.log10(stc)
	print("stc(log)",np.min(stc),np.max(stc))
	plt.imshow(stc[::-1,:],extent=[0,Lx/1000,-Lz/1000,0],
		zorder=100,alpha=0.2,cmap=plt.get_cmap("Greys"),vmin=-0.5,vmax=0.9)

	plt.text(100,10,"%.1lf Myr"%(tempo[0]/1.0E6))

	

	plt.savefig("LithoStrain_{:05}.png".format(cont*1), dpi=300)

	