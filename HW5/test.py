import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def neumann_1d_matrix(N, h):
    """
    Build 1D matrix for -d^2/dx^2 on nodes j=0..N (N+1 points),
    with homogeneous Neumann BCs using ghost points:
      u_{-1} = u_{1}, u_{N+1} = u_{N-1}
    """
    n = N + 1
    main = np.full(n, 2.0 / h**2)
    off = np.full(n - 1, -1.0 / h**2)

    T = sp.diags([off, main, off], offsets=[-1, 0, 1], format="lil")

    # Boundary rows modified by ghost-point elimination
    # j=0:   -u_xx ≈ (2u0 - 2u1)/h^2
    # j=N:   -u_xx ≈ (2uN - 2uN-1)/h^2
    T[0, 1] = -2.0 / h**2
    T[N, N - 1] = -2.0 / h**2

    return T.tocsr()


def build_operator_3d(N1, N2, N3):
    h1, h2, h3 = 1.0 / N1, 1.0 / N2, 1.0 / N3
    T1 = neumann_1d_matrix(N1, h1)
    T2 = neumann_1d_matrix(N2, h2)
    T3 = neumann_1d_matrix(N3, h3)

    I1 = sp.eye(N1 + 1, format="csr")
    I2 = sp.eye(N2 + 1, format="csr")
    I3 = sp.eye(N3 + 1, format="csr")

    # A = T1 ⊗ I2 ⊗ I3 + I1 ⊗ T2 ⊗ I3 + I1 ⊗ I2 ⊗ T3
    A = (
        sp.kron(sp.kron(T1, I2), I3, format="csr")
        + sp.kron(sp.kron(I1, T2), I3, format="csr")
        + sp.kron(sp.kron(I1, I2), T3, format="csr")
    )
    return A


def manufactured_solution(X, Y, Z):
    # Neumann-compatible exact solution
    u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
    f = 12 * np.pi**2 * u  # because -Δu = 12π^2 u
    return u, f


def solve_neumann_poisson_3d(N1=20, N2=20, N3=20):
    # Grid
    x = np.linspace(0.0, 1.0, N1 + 1)
    y = np.linspace(0.0, 1.0, N2 + 1)
    z = np.linspace(0.0, 1.0, N3 + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    u_exact, f = manufactured_solution(X, Y, Z)

    A = build_operator_3d(N1, N2, N3)
    b = f.ravel(order="C")

    n = b.size

    # Compatibility (continuous/discrete Neumann requires mean-zero RHS)
    vol_weight = (1.0 / N1) * (1.0 / N2) * (1.0 / N3)
    compat = b.sum() * vol_weight
    print(f"Discrete compatibility (approx integral of f): {compat:.3e}")

    # Augmented system to enforce mean(u)=0:
    # [A  1][u] = [b]
    # [1^T 0][c]   [0]
    ones = np.ones((n, 1))
    Aug = sp.bmat(
        [[A, sp.csr_matrix(ones)], [sp.csr_matrix(ones.T), None]],
        format="csr"
    )
    rhs = np.concatenate([b, np.array([0.0])])

    sol = spla.spsolve(Aug, rhs)
    u_num = sol[:n].reshape((N1 + 1, N2 + 1, N3 + 1), order="C")

    # Errors
    err = u_num - u_exact
    print(f"L_inf error: {np.max(np.abs(err)):.3e}")
    print(f"L2 error:    {np.sqrt(np.mean(err**2)):.3e}")

    # Plot a mid-plane slice z = 0.5
    k_mid = N3 // 2
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.contourf(x, y, u_exact[:, :, k_mid].T, levels=30)
    plt.colorbar()
    plt.title("Exact u(x,y,z_mid)")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 3, 2)
    plt.contourf(x, y, u_num[:, :, k_mid].T, levels=30)
    plt.colorbar()
    plt.title("Numerical u(x,y,z_mid)")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 3, 3)
    plt.contourf(x, y, err[:, :, k_mid].T, levels=30)
    plt.colorbar()
    plt.title("Error")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    solve_neumann_poisson_3d(N1=16, N2=16, N3=16)