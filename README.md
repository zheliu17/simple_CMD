# Simple CMD for 1D Systems

Author: Zhe Liu <zhe.liu2 [at] wisc.edu>

Ever tried to run a Centroid Molecular Dynamics (CMD) simulation for a simple 1D system, only to find the existing code was way too complex? I was in the same boat, so I wrote my own!

This project is a minimal, no-fuss implementation of CMD designed for any 1D potential. As a demonstration (`main.ipynb`), the code can reproduce the results in Figure 7 from the paper by Pérez, Tuckerman and Müser:

> *A comparative study of the centroid and ring-polymer molecular dynamics methods for approximating quantum time correlation functions from path integrals.* [J. Chem. Phys. 130, 184105 (2009)](https://doi.org/10.1063/1.3126950)

## What's inside?

This isn't just a CMD implementation. It also comes with a few other handy tools:

*   **DVR Method**: A Discrete Variable Representation (DVR) solver for the time-independent Schrödinger equation.
*   **Quantum Kubo Correlation Function**: Tools to compute quantum time correlation functions.
*   **MCMC Sampling**: A Metropolis Monte Carlo sampler to generate initial configurations for your 1D potential.

Hope you find it useful!