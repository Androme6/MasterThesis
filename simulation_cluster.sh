#!/bin/bash
#SBATCH --job-name=Three_modes_num    
#SBATCH --output=Three_modes_num_%j.txt     
#SBATCH --error=Three_modes_num_%j.txt       
#SBATCH --time=1:00:00                
#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=1            
#SBATCH --cpus-per-task=16             
#SBATCH --constraint=gpu               
#SBATCH --mem=0                     
#SBATCH --gpus=4         
#SBATCH --account=go072
                       

cd ~/MasterThesis

julia --project="." -e 'using Pkg; Pkg.instantiate()'

JULIA_NUM_THREADS=16 julia --project="." simulation.jl