using NCDatasets
using Glob
using CSV
using Printf
using DataFrames
using Base.Threads

# Converte os arquivos steps/step_<step>_<rank>.txt (partículas do Mandyoc) em _steps.nc
#
# Uso:
#   julia -t auto StepsNETCDF.jl [--strain] <pasta_do_modelo>
#
# Estrutura do _steps.nc (como o xarray enxerga):
#   x, z     (time, particle) float32  posição [m]; NaN quando a partícula não existe naquele passo
#   strain   (time, particle) float32  só com --strain (deformação acumulada, 5ª coluna)
#   layer    (particle)       int8     código da camada (constante para cada partícula)
#   particle (particle)       int32    id da partícula
#   time     (time)           float32  [Myr]
#   steps    (time)           int64
#   attrs: nx, nz, lx, lz, ncores
#
# A ideia é ler as partículas selecionadas com algo como:
#   ds = xr.open_dataset("_steps.nc")
#   ds.x.sel(particle=ids)   -> histórico completo de x das partículas escolhidas

# --- Funções Auxiliares (adaptadas de seus scripts) ---

function read_param(fpath::String="param.txt")::Dict{String,String}
    param_dict = Dict{String,String}()
    open(fpath, "r") do file
        for line in eachline(file)
            line=strip(line)
            if isempty(line) || startswith(line, "#")
                continue
            end
            line = split(line, "#")[1]
            line = replace(line, " " => "")
            key_value = split(line, "=")
            if length(key_value) == 2
                param_dict[lowercase(key_value[1])] = key_value[2]
            end
        end
    end
    return param_dict
end

function get_all_steps()::Vector{Int}
    # Usa os próprios arquivos de step (a pasta time/ pode já ter sido apagada)
    files = glob(joinpath("steps", "step_*_0.txt"))
    steps = [parse(Int, split(splitext(basename(f))[1], '_')[2]) for f in files]
    return sort(steps)
end

function read_time(step::Int)::Float32
    time::Float32=0.0
    open(joinpath("time", "time_$step.txt")) do file
        line = readline(file)
        line = split(line,"   ")[2]
        time = parse(Float32,line)/1e6 #Myr
    return time
    end
end

function get_times(steps::Vector{Int})::Vector{Float32}
    # Lê da pasta time/ se existir, senão do times.csv gerado pelo convertNETCDF_v2.jl
    if isdir("time")
        return Float32[read_time(step) for step in steps]
    elseif isfile("times.csv")
        df = CSV.read("times.csv", DataFrame)
        tdict = Dict(zip(df.step, Float32.(df.time_myr)))
        missing_steps = [s for s in steps if !haskey(tdict, s)]
        isempty(missing_steps) || @warn "Passos sem tempo no times.csv (time=NaN): $missing_steps"
        return Float32[get(tdict, s, NaN32) for s in steps]
    else
        error("Não encontrei a pasta time/ nem o times.csv")
    end
end

function read_step_file(fpath::String; read_strain::Bool=false)
    # Lê um arquivo de partículas e retorna x, z, id, layer (e strain)
    conf = CSV.File(fpath,
                    header=false,
                    comment="P",
                    delim=' ',
                    types=[Float32, Float32, Int32, Int8, Float32], # x, z, id, layer, strain
                    silencewarnings=true)

    strain = read_strain ? Vector(conf.Column5) : Float32[]
    return Vector(conf.Column1), Vector(conf.Column2), Vector(conf.Column3), Vector(conf.Column4), strain
end

function read_step(step::Int, ncores::Int; read_strain::Bool=false)
    # Junta as partículas de todos os ranks de um passo
    x = Float32[]; z = Float32[]; id = Int32[]; layer = Int8[]; strain = Float32[]
    for core in 0:(ncores-1)
        fpath = joinpath("steps", "step_$(step)_$core.txt")
        if isfile(fpath) && filesize(fpath) > 0
            xc, zc, idc, lc, sc = read_step_file(fpath, read_strain=read_strain)
            append!(x, xc); append!(z, zc); append!(id, idc); append!(layer, lc); append!(strain, sc)
        end
    end
    return (x=x, z=z, id=id, layer=layer, strain=strain)
end


function convert_steps_to_nc(pdict::Dict{String,Any})
    nc_fname = "_steps.nc"
    Nx = pdict["Nx"]
    Nz = pdict["Nz"]
    Lx = pdict["Lx"]
    Lz = pdict["Lz"]
    steps = pdict["steps"]
    num_steps = pdict["num_steps"]
    times = pdict["times"]
    ncores = pdict["ncores"]
    read_strain = pdict["strain"]
    xz_precision = pdict["xz_precision"]
    dfllevel::Int8 = 6 #compression level 1-9

    # --- 1. Leitura de todos os passos (em paralelo) ---
    data = Vector{Any}(undef, num_steps)
    progress_counter = Threads.Atomic{Int}(0)
    start_time = time()

    @threads for i in eachindex(steps)
        data[i] = read_step(steps[i], ncores, read_strain=read_strain)
        Threads.atomic_add!(progress_counter, 1)
        if progress_counter[] % 10 == 0
            @info "Reading progress: $(progress_counter[]) / $num_steps ($(round(progress_counter[]/num_steps*100, digits=1))%)"
        end
    end
    @info "Reading finished! Time: $(round((time() - start_time) / 60, digits=2)) minutes."

    # --- 2. Todas as partículas que aparecem em algum passo ---
    # As partículas só somem ao longo do tempo, mas usamos a união por segurança
    particle_ids = sort(unique(reduce(vcat, [d.id for d in data])))
    num_particles = length(particle_ids)
    println("Total de partículas distintas: $num_particles")

    # Camada de cada partícula (fixa ao longo do tempo)
    particle_layer = fill(Int8(-1), num_particles)
    n_layer_changes = 0
    for d in data
        idx = searchsortedfirst.(Ref(particle_ids), d.id)
        for k in eachindex(idx)
            if particle_layer[idx[k]] == -1
                particle_layer[idx[k]] = d.layer[k]
            elseif particle_layer[idx[k]] != d.layer[k]
                n_layer_changes += 1
            end
        end
    end
    n_layer_changes == 0 || @warn "$n_layer_changes registros com camada diferente da primeira aparição da partícula"

    # --- 3. Escrita no NetCDF, um passo de cada vez (não precisa da matriz inteira na memória) ---
    Dataset(nc_fname, "c") do ds
        defDim(ds, "particle", num_particles)
        defDim(ds, "time", num_steps)

        defVar(ds, "time", Float32.(times), ("time",),
            attrib=Dict("units"=>"Myr", "long_name"=>"Model time", "axis"=>"T"),
            deflatelevel=dfllevel, shuffle=true)
        defVar(ds, "steps", steps, ("time",),
            attrib=Dict("long_name"=>"Steps"),
            deflatelevel=dfllevel, shuffle=true)
        defVar(ds, "particle", particle_ids, ("particle",),
            attrib=Dict("long_name"=>"Particle id"),
            deflatelevel=dfllevel, shuffle=true)
        defVar(ds, "layer", particle_layer, ("particle",),
            attrib=Dict("long_name"=>"Layer code", "comment"=>"Order of the layers in interfaces.txt (0 = bottom layer)"),
            deflatelevel=dfllevel, shuffle=true)

        # chunk = todas as partículas de um passo (ler um passo inteiro é barato)
        chunks = (num_particles, 1)
        vars = [("x", "x-coordinate"), ("z", "z-coordinate")]
        read_strain && push!(vars, ("strain", "Accumulated strain"))
        for (name, longname) in vars
            units = name == "strain" ? "dimensionless" : "m"
            defVar(ds, name, Float32, ("particle", "time"),
                attrib=Dict("units"=>units, "long_name"=>longname, "_FillValue"=>NaN32),
                chunksizes=chunks, deflatelevel=dfllevel, shuffle=true)
        end

        column = fill(NaN32, num_particles)
        for i in eachindex(steps)
            d = data[i]
            idx = searchsortedfirst.(Ref(particle_ids), d.id)

            # Arredondar para xz_precision [m] ajuda muito a compressão
            # (os step*.txt já têm só 6 algarismos significativos)
            fill!(column, NaN32); column[idx] = round.(d.x ./ xz_precision) .* xz_precision
            ds["x"][:, i] = column
            fill!(column, NaN32); column[idx] = round.(d.z ./ xz_precision) .* xz_precision
            ds["z"][:, i] = column
            if read_strain
                fill!(column, NaN32); column[idx] = d.strain
                ds["strain"][:, i] = column
            end
            data[i] = nothing # libera memória
        end

        ds.attrib["nx"] = Nx
        ds.attrib["nz"] = Nz
        ds.attrib["lx"] = Lx
        ds.attrib["lz"] = Lz
        ds.attrib["ncores"] = ncores
        ds.attrib["xz_precision_m"] = xz_precision
        println("\nSaved to $nc_fname")
    end
    @info "Finished! Total time: $(round((time() - start_time) / 60, digits=2)) minutes."
end

# --- Lógica Principal do Script ---

read_strain = "--strain" in ARGS # inclui a deformação acumulada (deixa o arquivo mais pesado)
data_dir = ARGS[end]
cd(data_dir)

params = read_param("param.txt")
Nx = parse(Int, params["nx"])
Nz = parse(Int, params["nz"])
Lx = parse(Float32, params["lx"])
Lz = parse(Float32, params["lz"])

steps::Vector{Int} = get_all_steps()
num_steps::Int = length(steps)
times = get_times(steps)
ncores::Int = size(glob(joinpath("steps","step_0_*.txt")))[1]
println("Encontrados $num_steps passos de tempo, de $(steps[1]) a $(steps[end]).")
println("Número de núcleos detectados: $ncores")
println("Threads do Julia: $(Threads.nthreads())")

pdict = Dict{String,Any}(
    "Nx"=>Nx, "Nz"=>Nz, "Lx"=>Lx, "Lz"=>Lz,
    "steps"=>steps, "num_steps"=>num_steps, "times"=>times,
    "ncores"=>ncores, "strain"=>read_strain,
    "xz_precision"=>1.0f0 # [m]
)

println("Convertendo arquivos de partículas...")
convert_steps_to_nc(pdict)
println("\nConversão concluída!")
