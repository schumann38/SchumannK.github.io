import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# IE 310/317 Markov Chains Project
# Code aligned with the final report's CU definition:
# A CU is counted only in neighboring 2x2 windows of the matrix.
# ============================================================

# Original Finch Species Co-occurrence Matrix
M0 = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # A
    [1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1],  # B
    [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0],  # C
    [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],  # D
    [1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],  # E
    [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],  # F
    [1,1,1,1,1,0,1,1,0,0,0,0,0,1,0,1,1],  # G
    [1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0],  # H
    [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],  # I
    [1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],  # J
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],  # K
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # L
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]   # M
], dtype=int)


def CUcount(M):
    """
    Counts checkerboard units exactly as described in the final report:
    only adjacent 2x2 blocks are checked.

    A checkerboard CU is:
        1 0      0 1
        0 1  or  1 0
    """
    count = 0
    rows, cols = M.shape

    for i in range(rows - 1):
        for j in range(cols - 1):
            block = M[i:i+2, j:j+2]

            pattern1 = np.array([[1, 0],
                                 [0, 1]])
            pattern2 = np.array([[0, 1],
                                 [1, 0]])

            if np.array_equal(block, pattern1) or np.array_equal(block, pattern2):
                count += 1

    return count


def markov_step(M, rng):
    """
    Performs one step of the swap-based Markov chain from the project:
    1. Randomly choose two different rows.
    2. Randomly choose two different columns.
    3. If the selected 2x2 submatrix is a checkerboard, swap it.
    4. Otherwise, do nothing.
    """
    rows, cols = M.shape

    i, j = rng.choice(rows, size=2, replace=False)
    a, b = rng.choice(cols, size=2, replace=False)

    sub = M[np.ix_([i, j], [a, b])]

    pattern1 = np.array([[1, 0],
                         [0, 1]])
    pattern2 = np.array([[0, 1],
                         [1, 0]])

    if np.array_equal(sub, pattern1):
        M[np.ix_([i, j], [a, b])] = pattern2
    elif np.array_equal(sub, pattern2):
        M[np.ix_([i, j], [a, b])] = pattern1

    return M


def run_simulation(M_initial, steps=10000, seed=123):
    """
    Simulates the Markov chain and records C_n = CUcount(X_n).

    The final report does not list its exact random seed, so the statistics
    may not match digit-for-digit unless the same seed/random sequence is used.
    This code matches the report's method and counting convention.
    """
    rng = np.random.default_rng(seed)
    M = M_initial.copy()

    C_values = [CUcount(M)]

    for _ in range(steps):
        M = markov_step(M, rng)
        C_values.append(CUcount(M))

    return np.array(C_values)


def print_statistics(C_values):
    """
    Prints the same main statistics shown in the final report.
    """
    print("— Part 3 Statistics —")
    print(f"C_0 (original matrix CU count): {C_values[0]}")
    print(f"Sample mean E[C_n]        : {np.mean(C_values):.4f}")
    print(f"Sample std  s[C_n]        : {np.std(C_values, ddof=1):.4f}")
    print(f"Min C_n                   : {np.min(C_values)}")
    print(f"Max C_n                   : {np.max(C_values)}")


def make_report_plot(C_values, output_file="part3_simulation.png"):
    """
    Creates a figure similar to the final report:
    - Left: C_n vs. n
    - Right: Histogram of C_n
    """
    mean_c = np.mean(C_values)
    original_c = C_values[0]
    steps = len(C_values) - 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left plot: C_n versus n
    axes[0].plot(range(len(C_values)), C_values, linewidth=1, label=r"$C_n$")
    axes[0].axhline(mean_c, linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_c:.2f}")
    axes[0].axhline(original_c, linestyle=":", linewidth=1.5,
                    label=f"$C_0$ = {original_c}")
    axes[0].set_title(r"$C_n$ vs. $n$")
    axes[0].set_xlabel("Step n")
    axes[0].set_ylabel(r"$C_n$ = CUcount($X_n$)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Right plot: histogram
    bins = np.arange(np.min(C_values) - 0.5, np.max(C_values) + 1.5, 1)
    axes[1].hist(C_values, bins=bins, density=True, alpha=0.75, edgecolor="black")
    axes[1].axvline(mean_c, linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_c:.2f}")
    axes[1].axvline(original_c, linestyle=":", linewidth=1.5,
                    label=f"$C_0$ = {original_c}")
    axes[1].set_title(r"Histogram of $C_n$")
    axes[1].set_xlabel("CU Count")
    axes[1].set_ylabel("Relative Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Part 3: Markov Chain Simulation of CU Counts", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nPlot saved as '{output_file}'")


# ============================================================
# Run the analysis
# ============================================================
if __name__ == "__main__":
    C_values = run_simulation(M0, steps=10000, seed=123)

    print_statistics(C_values)
    make_report_plot(C_values, output_file="part3_simulation.png")
