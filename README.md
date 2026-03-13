# APMA-E4301

Numerical methods and PDE homework code for APMA E4301.

## Repository Contents

### HW1
- `HW1/HW1b.py`
  - Forward Euler for the logistic ODE: y' = y(1-y)
  - Compares numerical solutions across multiple time steps
  - Computes max error vs exact solution and observed convergence order
- `HW1/HW1.2b .py`
  - Backward Euler for the logistic ODE
  - Solves the implicit update with the quadratic closed-form root
  - Compares max errors and observed convergence order across step sizes

### HW04
- `HW04/Q1.py`
  - 1D convection-diffusion IBVP schemes with Dirichlet boundaries
  - Implements and compares:
    - Scheme (a): explicit convection + explicit diffusion
    - Scheme (b): explicit convection + implicit diffusion
    - Scheme (c): implicit convection + implicit diffusion
  - Includes stability checks using lambda and mu
  - Produces final-time solution plots after 80 steps

### HW5
- `HW5/ex2.py`
  - Forward Euler for the heat equation u_t = u_xx on [0,1]
  - Uses homogeneous Neumann boundary conditions via ghost points
  - Produces 3D surface plots for selected time-step sizes
- `HW5/test.py`
  - 3D Poisson solver with homogeneous Neumann boundary conditions
  - Uses sparse Kronecker-sum operator assembly
  - Uses an augmented linear system to enforce mean(u)=0
  - Compares numerical and manufactured exact solutions and reports errors

## Dependencies

Main Python libraries used:
- numpy
- matplotlib
- scipy (for sparse linear algebra in HW5/test.py)

Install with:

pip install numpy matplotlib scipy

## Notes

- Scripts are currently standalone and can be run individually.
- Some files generate plots directly and print diagnostics to terminal.
