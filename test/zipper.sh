#!/usr/bin/env bash
DIRNAME=${PWD##*/}
# Primeiro zipa os arquivos fixos
zip "$DIRNAME.zip" interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh
# Lista de padrões
patterns=(
"bc_velocity_*.txt"
"density_*.txt"
"heat_*.txt"
"pressure_*.txt"
"sp_surface_global_*.txt"
"strain_*.txt"
"temperature_*.txt"
"time_*.txt"
"velocity_*.txt"
"viscosity_*.txt"
"scale_bcv.txt"
"step*.txt"
"Phi*.txt"
"dPhi*.txt"
"X_depletion*.txt"
"*.bin*.txt"
"bc*-1.txt"
"*.log"
)
# Faz um loop e usa find para evitar o erro "argument list too long"
for pat in "${patterns[@]}"; do
find . -maxdepth 1 -type f -name "$pat" -exec zip -u "$DIRNAME.zip" {} +
done
