# Cat States in Ultrastrongly Coupled Circuit QED

Code and simulation data for my Master's thesis at EPFL.

## Overview
This project investigates whether it is possible to generate Schrödinger cat states, quantum superpositions of coherent states with significant applications in quantum technologies, in a circuit Quantum Electrodynamics architecture. The specific system under study consists of a multimode resonator ultrastrongly coupled to a superconducting qubit.

The codebase is built to simulate both the full and the effective driven-dissipative setups from first principles.

## Prerequisites
The simulations are written in [Julia](https://julialang.org/) and rely heavily on the [QuantumToolbox.jl](https://github.com/QuantumToolbox/QuantumToolbox.jl) framework.

To run this code, you will need:
* Julia v1.x (or higher)
* `QuantumToolbox`
* `CairoMakie` (for plotting)
* `CUDA` (if running on GPU)
* `JLD2` (for data saving/loading)

## Setup and Installation
Clone this repository and instantiate the Julia environment to automatically install all dependencies:

```bash
git clone https://github.com/Androme6/MasterThesis.git
cd MasterThesis
julia --project=. -e 'using Pkg; Pkg.instantiate()'
