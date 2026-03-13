import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def run_scheme(k, N=20, Tfinal=0.5):
    """Run forward Euler scheme for u_t = u_xx with Neumann BC.

    Parameters
    ----------
    k : float
        Time step size.
    N : int, optional
        Number of spatial intervals (default 20, so h = 1/N).
    Tfinal : float, optional
        Final time (default 0.5).
    """
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)  # j = 0..N
    mu = k / h**2

    Nt = int(round(Tfinal / k))
    t = np.linspace(0.0, Tfinal, Nt + 1)

    # initial condition u0(x) = cos^2(pi x / 2)
    u = np.cos(0.5 * np.pi * x) ** 2
    U = np.zeros((Nt + 1, N + 1))
    U[0, :] = u

    for n in range(Nt):
        u_new = u.copy()

        # interior points j = 1..N-1
        for j in range(1, N):
            u_new[j] = mu * u[j - 1] + (1 - 2 * mu) * u[j] + mu * u[j + 1]

        # Neumann BC using ghost points: u_{-1} = u_1, u_{N+1} = u_{N-1}
        u_new[0] = (1 - 2 * mu) * u[0] + 2 * mu * u[1]      # j = 0
        u_new[N] = 2 * mu * u[N - 1] + (1 - 2 * mu) * u[N]  # j = N

        u = u_new
        U[n + 1, :] = u

    return x, t, U, mu


def plot_surface(x, t, U, k, mu):
    X, T = np.meshgrid(x, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, T, U, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    ax.set_title(f"Forward Euler, N=20, T=0.5, k={k}, mu={mu:.3f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N = 20
    Tfinal = 0.5

    for k in [0.005, 0.0005]:
        x, t, U, mu = run_scheme(k, N=N, Tfinal=Tfinal)
        print(f"k = {k}, mu = {mu}")
        plot_surface(x, t, U, k, mu)