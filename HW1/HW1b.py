import numpy as np
import matplotlib.pyplot as plt

# We consider y' = y(1-y) with y(0) in (0,1)

# Exact solution of the logistic equation for comparison
def exact_solution(t, y0):
    C = 1 / y0 - 1
    return 1 / (1 + C * np.exp(-t))


# CODE FOR EULER FORWARD METHOD
def ForwardEulerStep(T, y0, h):
    N = int(T/h)
    t0 = np.linspace(0, T, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for i in range (0, len(t0)-1):
        y[i+1] = y[i] + h * (y[i] * (1 - y[i]))
    return y


# TESTING THE FUNCTION AND PLOTTING FOR DIFFERENT h
if __name__ == "__main__":
    y0 = 0.2
    T = 10
    h_values = [0.025, 0.1, 0.4, 1.6, 3.2]

    # larger figure so curves are easier to distinguish
    plt.figure(figsize=(10, 6))

    # different markers/linestyles for each h
    markers = ['o', 's', '^', 'D', 'x']
    linestyles = ['-', '--', '-.', ':', '-']

    max_errors = []

    for idx, h in enumerate(h_values):
        y = ForwardEulerStep(T, y0, h)
        t = np.linspace(0, T, int(T/h) + 1)

        # plot numerical solution
        plt.plot(t, y,
                 label=f'h = {h}',
                 marker=markers[idx],
                 linestyle=linestyles[idx],
                 markersize=4,
                 linewidth=1.5)

        # compute max error against exact solution on this grid
        y_exact = exact_solution(t, y0)
        max_err = np.max(np.abs(y - y_exact))
        max_errors.append(max_err)

    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("Forward Euler solutions for different step sizes, y(0) = 0.2, t ∈ [0,10]")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot max error versus h on a standard (non-log) scale
    plt.figure(figsize=(8, 5))
    h_array = np.array(h_values, dtype=float)
    errors_array = np.array(max_errors, dtype=float)

    plt.plot(h_array, errors_array, 'o-', label='max error')
    plt.xlabel('step size h')
    plt.ylabel('max error on [0,10]')
    plt.title('Convergence of Forward Euler (max error vs h)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # print observed orders of convergence between consecutive h values
    orders = np.log(errors_array[1:] / errors_array[:-1]) / np.log(h_array[1:] / h_array[:-1])
    print("h values:", h_values)
    print("max errors:", max_errors)
    print("observed orders:", orders)