include("setup_gap.jl")

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

kp = 0.1 
tmax = 15000
t_selected = tmax
nframes = 500

###########
H_fun = H_ideal
filename = "Ideal"
###########

timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
save_dir = "C:\\Users\\andre\\Desktop\\Università\\Magistrale\\MA4\\Thesis\\Code\\MasterThesis\\gap"*timestamp
#save_dir = "/capstor/store/cscs/2go/go072/alanteri/gap"+timestamp

F_list = range(0.005, 0.025, length=10)

gap_output = gap_finder(params, H_fun, F_list, kp, tmax, nframes, filename, save_dir)
display(gap_output[3])
