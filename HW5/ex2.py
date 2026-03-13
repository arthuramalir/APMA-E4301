import numpy as np
import matplotlib.pyplot as plt

"""
Problem 1. (6 points)Let ν > 0, r > 0 be given constants and f(x) be a given function. Consider
the numerical solution of the IBVP ut + rux = νuxx for x ∈ [0,1] and t > 0 with Dirichlet BCs
u(t, 0)=u(t,1)=0 and IC u(0,x)=f(x) on a uniform spatial grid {xj=jh= j
N} in space. Find the con
ditions, if any, on λ=rk
h and/or µ=νk
h2
to assure the L∞ stability of the following discretizations of the
IBVP respectively:
a) If the upwind difference for convection and the 2nd order difference for diffusion are used,
together with the Forward Euler in-time, i.e., Uh
n+1=Uh
n − rkD−
hUh
n + νkD2
hUh
n.
b) Like a), but with Backward Euler in-time for diffusion, and forward Euler in-time for convection,
i.e. Uh
n+1=Uh
n − rkD−
hUh
n + νkD2
hUh
n+1.
c) Like a), but with Backward Euler in-time for both diffusion and convection, i.e.
Uh
n+1=Uh
n − rkD−
hUh
n+1 + νkD2
hUh
n+1.
Problem 2. (6 points) For the equations in Prob 1, let f(x)=max{1−8|x−0.25|,0}, r=1 and ν=0.002.
Take N=128 and µ=0.2, carry out numerical experiments, and present, respectively, the plots of the
numerical solution at the end of 80 time steps, by using respectively each of the three discretization
schemes given in Problem 1. Compare/explain the results with those for a), and check if they match
the theory
"""


def f(x):
    return np.maximum(1.0 - 8.0 * np.abs(x - 0.25), 0.0)


def initial_condition(N):
    x = np.linspace(0.0, 1.0, N + 1)
    u0 = f(x)
    u0[0] = 0.0
    u0[-1] = 0.0
    return x, u0


def stability_report(lam, mu):
    a_ok = (mu >= 0.0) and (lam >= 0.0) and (lam + 2.0 * mu <= 1.0)
    b_ok = (lam >= 0.0) and (lam <= 1.0) and (mu >= 0.0)
    c_ok = (lam >= 0.0) and (mu >= 0.0)
    return a_ok, b_ok, c_ok


def scheme_a(N, r, nu, k, nsteps):
    h = 1.0 / N
    lam = r * k / h
    mu = nu * k / h**2

    x, u = initial_condition(N)

    for _ in range(nsteps):
        u_new = u.copy()
        u_new[1:N] = (
            u[1:N]
            - lam * (u[1:N] - u[0:N - 1])
            + mu * (u[2:N + 1] - 2.0 * u[1:N] + u[0:N - 1])
        )
        u_new[0] = 0.0
        u_new[N] = 0.0
        u = u_new
    return x, u


def scheme_b(N, r, nu, k, nsteps):
    h = 1.0 / N
    lam = r * k / h
    mu = nu * k / h**2

    x, u = initial_condition(N)

    m = N - 1
    main_diag = (1.0 + 2.0 * mu) * np.ones(m)
    off_diag = -mu * np.ones(m - 1)
    A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)

    for _ in range(nsteps):
        rhs = u.copy()
        rhs[1:N] = u[1:N] - lam * (u[1:N] - u[0:N - 1])

        u_int = np.linalg.solve(A, rhs[1:N])
        u[1:N] = u_int
        u[0] = 0.0
        u[N] = 0.0
    return x, u


def scheme_c(N, r, nu, k, nsteps):
    h = 1.0 / N
    lam = r * k / h
    mu = nu * k / h**2

    x, u = initial_condition(N)

    m = N - 1
    main_diag = (1.0 + lam + 2.0 * mu) * np.ones(m)
    lower_diag = -(lam + mu) * np.ones(m - 1)
    upper_diag = -mu * np.ones(m - 1)
    A = np.diag(main_diag) + np.diag(lower_diag, -1) + np.diag(upper_diag, 1)

    for _ in range(nsteps):
        u_int = np.linalg.solve(A, u[1:N])
        u[1:N] = u_int
        u[0] = 0.0
        u[N] = 0.0

    return x, u


def run_problem_2():
    r = 1.0
    nu = 0.002
    N = 128
    mu_target = 0.2

    h = 1.0 / N
    k = mu_target * h**2 / nu
    nsteps = 80

    lam = r * k / h
    mu = nu * k / h**2
    a_ok, b_ok, c_ok = stability_report(lam, mu)
    print(f"lambda={lam:.6f}, mu={mu:.6f}")
    print(f"Scheme (a) stable condition met? {a_ok}")
    print(f"Scheme (b) stable condition met? {b_ok}")
    print(f"Scheme (c) stable condition met? {c_ok}")

    x_a, u_a = scheme_a(N, r, nu, k, nsteps)
    x_b, u_b = scheme_b(N, r, nu, k, nsteps)
    x_c, u_c = scheme_c(N, r, nu, k, nsteps)

    # Combined plot (kept)
    plt.figure(figsize=(8, 5))
    plt.plot(x_a, u_a, label="(a) explicit FE")
    plt.plot(x_b, u_b, label="(b) semi-implicit")
    plt.plot(x_c, u_c, label="(c) fully implicit")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title("Problem 2: N=128, mu=0.2, nsteps=80 (all schemes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Separate plot: scheme (a)
    plt.figure(figsize=(8, 5))
    plt.plot(x_a, u_a, color="tab:blue")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title("Scheme (a): explicit FE")
    plt.grid(True)
    plt.tight_layout()

    # Separate plot: scheme (b)
    plt.figure(figsize=(8, 5))
    plt.plot(x_b, u_b, color="tab:orange")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title("Scheme (b): semi-implicit")
    plt.grid(True)
    plt.tight_layout()

    # Separate plot: scheme (c)
    plt.figure(figsize=(8, 5))
    plt.plot(x_c, u_c, color="tab:green")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title("Scheme (c): fully implicit")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_problem_2()