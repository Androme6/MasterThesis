include("setup_simulation.jl")

# 1. Set parameters 
params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.0, 
    ωp = 0.0, 
    ωq = 2.5, 
    g1 = 0.1, 
    g2 = 0.2,   
    g2p = 0, 
    θ = π / 6.0,
    ωd = 0.0
)

F = 0.025
kp = 0.1 
tmax = 15000
t_selected = tmax
nframes = 500

###########
H_fun = H_eff_RWA
filename = "RWA"
###########

save_dir = "C:\\Users\\andre\\Desktop\\Università\\Magistrale\\MA4\\Thesis\\Code\\MasterThesis\\Output"
#save_dir = "/capstor/store/cscs/2go/go072/alanteri"

output = run_simulation(params, H_fun, filename, F, kp, tmax, t_selected, nframes, save_dir)