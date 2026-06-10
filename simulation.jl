include("setup_simulation.jl")

# 1. Set parameters 
params = SystemParams(
    ω1 = 5.0,  
    θ = π / 6.0,
    F = 0.025
)

tmax = 1000
t_selected = tmax
nframes = 500

###########
H_fun = H_full
filename = "AAAAAAAAA"
###########

save_dir = "C:\\Users\\andre\\Desktop\\Università\\Magistrale\\MA4\\Thesis\\Code\\MasterThesis\\Output"
#save_dir = "/capstor/store/cscs/2go/go072/alanteri"

output = run_simulation(params, H_fun, filename, tmax, t_selected, nframes, save_dir, true, false)