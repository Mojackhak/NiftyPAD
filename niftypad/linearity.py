import numpy as np
import matplotlib.pyplot as plt

def logan_linear_phase(C_t, C_ref, t, t_trunc=(200, np.inf), save_dir=None):
    """
    Identify the linear phase for Logan graphical analysis.

    Logan Graphical Model Equations:
    ∫₀ᵀ C_T(t)dt / C_T(T) = DVR * ∫₀ᵀ C_R(t)dt / C_T(T) + int'

    where:
    - int' stands for the intercept in the linear regression,
    - DVR is the distribution volume ratio,
    - BP_ND = DVR - 1

    Parameters:
        C_t (np.array): Time-activity curve (TAC) of the target region.
        C_ref (np.array): Time-activity curve (TAC) of the reference region.
        t (np.array): Time points corresponding to the TAC values.
        t_trunc (tuple): Tuple specifying the time range to plot (default: (200, np.inf)).
        save_dir (str, optional): Directory to save the plot as an SVG file. Default is None.

    Returns:
        slopes (list): Slopes between adjacent points.
    """
    # Ensure np array
    C_t = np.array(C_t)
    C_ref = np.array(C_ref)
    t = np.array(t)

    # Calculate integrals using trapezoidal rule
    integral_Ct = np.array([np.trapz(C_t[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Ct = np.insert(integral_Ct, 0, 0)  # Insert 0 at the beginning for alignment
    integral_Cref = np.array([np.trapz(C_ref[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Cref = np.insert(integral_Cref, 0, 0)  # Insert 0 at the beginning for alignment

    # mask the data
    mask = (t >= t_trunc[0]) & (t <= t_trunc[1])
    t = t[mask]
    C_t = C_t[mask]
    C_ref = C_ref[mask]
    integral_Ct = integral_Ct[mask]
    integral_Cref = integral_Cref[mask]


    # Normalize by instantaneous activity
    Logan_y = integral_Ct / C_t
    Logan_x = integral_Cref / C_t

    # Calculate slopes between adjacent points
    slopes = np.gradient(Logan_y) / np.gradient(Logan_x)

    # Visualize Logan plot with slopes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel("$\\int_0^t C_{{ref}}(t)dt / C_t(t)$")
    ax1.set_ylabel("$\\int_0^t C_t(t)dt / C_t(t)$", color=color)
    ax1.plot(Logan_x, Logan_y, '-', label='Logan Data', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Slopes', color=color)  # we already handled the x-label with ax1
    ax2.plot(Logan_x, slopes, 'r-', label='Slopes', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Logan Plot with Slopes")

    # Save the plot if save_dir is provided
    if save_dir is not None:
        file_path = f"{save_dir}/logan_plot_with_slopes.svg"
        plt.savefig(file_path, format='svg')
        print(f"Plot saved to {file_path}")

    plt.show()


def logan_k2p_linear_phase(C_t, C_ref, t, k2p, t_trunc=(200, np.inf), save_dir=None):
    """
    Identify the linear phase for Logan2 graphical analysis with pre-defined k2p.

    Logan2 Graphical Model Equations:
    ∫₀ᵀ C_T(t)dt / C_T(T) = DVR * ∫₀ᵀ C_R(t)dt / C_T(T) + C_R(t) / k₂' + int'

    where:
    - int' stands for the intercept in the linear regression,
    - DVR is the distribution volume ratio,
    - BP_ND = DVR - 1

    Parameters:
        C_t (np.array): Time-activity curve (TAC) of the target region.
        C_ref (np.array): Time-activity curve (TAC) of the reference region.
        t (np.array): Time points corresponding to the TAC values.
        k2p (float): Pre-defined k₂' value.
        t_trunc (tuple): Tuple specifying the time range to plot (default: (200, np.inf)).
        save_dir (str, optional): Directory to save the plot as an SVG file. Default is None.

    Returns:
        None
    """
    # Ensure np array
    C_t = np.array(C_t)
    C_ref = np.array(C_ref)
    t = np.array(t)

    # Calculate integrals using trapezoidal rule
    integral_Ct = np.array([np.trapz(C_t[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Ct = np.insert(integral_Ct, 0, 0)  # Insert 0 at the beginning for alignment
    integral_Cref = np.array([np.trapz(C_ref[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Cref = np.insert(integral_Cref, 0, 0)  # Insert 0 at the beginning for alignment

    # mask the data
    mask = (t >= t_trunc[0]) & (t <= t_trunc[1])
    t = t[mask]
    C_t = C_t[mask]
    C_ref = C_ref[mask]
    integral_Ct = integral_Ct[mask]
    integral_Cref = integral_Cref[mask]

    # Calculate normalized variables
    Logan_y = integral_Ct / C_t - C_ref / k2p
    Logan_x = integral_Cref / C_t

    # Calculate slopes between adjacent points
    slopes = np.gradient(Logan_y) / np.gradient(Logan_x)

    # Visualize Logan plot with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: Logan plot with normalized variables
    ax1 = axes[0]
    ax1.plot(Logan_x, Logan_y, label="$\\int_0^T C_T(t)dt / C_T(T) - C_R(T) / k_2'$", color="tab:blue")
    ax1.set_xlabel("$\\int_0^T C_R(t)dt / C_T(T)$")
    ax1.set_ylabel("$\\int_0^T C_T(t)dt / C_T(T) - C_R(T) / k_2'$")

    # Add slopes to the first subplot
    ax2 = ax1.twinx()
    ax2.plot(Logan_x, slopes, label="Slopes", color="tab:red")
    ax2.set_ylabel("Slopes", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # Combine legends for both axes in the first subplot
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    # Second subplot: Time vs normalized integral_Cref
    axes[1].plot(t, integral_Cref / C_t, label="$t$ vs $\\int_0^T C_R(t)dt / C_T(T)$", color="tab:green")
    axes[1].set_xlabel("Time (t)")
    axes[1].set_ylabel("$\\int_0^T C_R(t)dt / C_T(T)$")
    axes[1].legend()

    # Adjust layout
    fig.tight_layout()

    # Save the plot if save_dir is provided
    if save_dir is not None:
        file_path = f"{save_dir}/logan_k2p_linear_phase.svg"
        plt.savefig(file_path, format='svg')
        print(f"Plot saved to {file_path}")

    plt.show()



def mrtm_linear_phase(C_t, C_ref, t, t_trunc=(200, np.inf), save_dir=None):
    """
    Plot MRTM linear phase with subplots.

    MRTM Equation:
        C_T(T) = γ₁ * ∫₀ᵀ C_R(t)dt + γ₂ * ∫₀ᵀ C_T(t)dt + γ₃ * C_R(T)

    Binding Potential Calculation:
        BP_ND = -(γ₁ / γ₂ + 1)

    Parameters:
        C_t (np.array): Time-activity curve (TAC) of the target region.
        C_ref (np.array): Time-activity curve (TAC) of the reference region.
        t (np.array): Time points corresponding to the TAC values.
        t_trunc (tuple): Tuple specifying the time range to plot (default: (200, np.inf)).
        save_dir (str, optional): Directory to save the plot as an SVG file. Default is None.

    Returns:
        None
    """
    # Ensure np array
    C_t = np.array(C_t)
    C_ref = np.array(C_ref)
    t = np.array(t)

    # Calculate integrals using trapezoidal rule
    integral_Ct = np.array([np.trapz(C_t[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Ct = np.insert(integral_Ct, 0, 0)  # Insert 0 at the beginning for alignment
    integral_Cref = np.array([np.trapz(C_ref[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Cref = np.insert(integral_Cref, 0, 0)  # Insert 0 at the beginning for alignment

    # mask the data
    mask = (t >= t_trunc[0]) & (t <= t_trunc[1])
    t = t[mask]
    C_t = C_t[mask]
    C_ref = C_ref[mask]
    integral_Ct = integral_Ct[mask]
    integral_Cref = integral_Cref[mask]

    # Prepare data for plotting
    slopes_Ct = np.gradient(C_t / C_ref) / np.gradient(integral_Ct / C_ref)
    slopes_Cref = np.gradient(C_t / C_ref) / np.gradient(integral_Cref / C_ref)

    # Visualize MRTM plot with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # First row: plots with x-axis as normalized integrals or reference TAC
    axes[0, 0].plot(integral_Cref / C_ref, C_t / C_ref, label="$\\int_0^T C_{\\text{R}}(t)dt / C_{\\text{R}}(T)$ vs $C_T(T) / C_{\\text{R}}(T)$", color="tab:blue")
    axes[0, 0].set_xlabel("$\\int_0^T C_{\\text{R}}(t)dt / C_{\\text{R}}(T)$")
    axes[0, 0].set_ylabel("$C_T(T) / C_{\\text{R}}(T)$")

    axes[0, 1].plot(integral_Ct / C_ref, C_t / C_ref, label="$\\int_0^T C_T(t)dt / C_{\\text{R}}(T)$ vs $C_T(T) / C_{\\text{R}}(T)$", color="tab:green")
    axes[0, 1].set_xlabel("$\\int_0^T C_T(t)dt / C_{\\text{R}}(T)$")
    axes[0, 1].set_ylabel("$C_T(T) / C_{\\text{R}}(T)$")

    axes[0, 2].plot(t, integral_Cref / C_ref, label="$t$ vs $\\int_0^T C_{\\text{R}}(t)dt / C_{\\text{R}}(T)$", color="tab:orange")
    axes[0, 2].set_xlabel("$t$")
    axes[0, 2].set_ylabel("$\\int_0^T C_{\\text{R}}(t)dt / C_{\\text{R}}(T)$")

    # Second row: slopes with corresponding x-axis
    axes[1, 0].plot(integral_Cref / C_ref, slopes_Cref, label="Slope: $\\int_0^T C_{\\text{R}}(t)dt / C_{\\text{R}}(T)$", color="tab:blue")
    axes[1, 0].set_xlabel("$\\int_0^T C_{\\text{R}}(t)dt / C_{\\text{R}}(T)$")
    axes[1, 0].set_ylabel("Slope")

    axes[1, 1].plot(integral_Ct / C_ref, slopes_Ct, label="Slope: $\\int_0^T C_T(t)dt / C_{\\text{R}}(T)$", color="tab:green")
    axes[1, 1].set_xlabel("$\\int_0^T C_T(t)dt / C_{\\text{R}}(T)$")
    axes[1, 1].set_ylabel("Slope")

    axes[1, 2].axis("off")  # Leave the last subplot blank

    # Adjust layout
    for ax in axes.flat:
        if ax.get_legend():
            ax.legend()
        ax.grid()

    fig.tight_layout()

    # Save the plot if save_dir is provided
    if save_dir is not None:
        file_path = f"{save_dir}/mrtm_subplots_with_slopes.svg"
        plt.savefig(file_path, format='svg')
        print(f"Plot saved to {file_path}")

    plt.show()



def mrtm_k2p_linear_phase(C_t, C_ref, t, k2p, t_trunc=(200, np.inf), save_dir=None):
    """
    Plot MRTM2 (MRTM with pre-defined k2p) linear phase with subplots.

    MRTM2 Equation:
        C_T(T) = γ₁ * (∫₀ᵀ C_R(t)dt + C_R(T) / k₂') + γ₂ * ∫₀ᵀ C_T(t)dt 

    Binding Potential Calculation:
        BP_ND = -(γ₁ / γ₂ + 1)

    Parameters:
        C_t (np.array): Time-activity curve (TAC) of the target region.
        C_ref (np.array): Time-activity curve (TAC) of the reference region.
        t (np.array): Time points corresponding to the TAC values.
        k2p (float): Pre-defined k₂' value.
        t_trunc (tuple): Tuple specifying the time range to plot (default: (200, np.inf)).
        save_dir (str, optional): Directory to save the plot as an SVG file. Default is None.

    Returns:
        None
    """
    # Ensure np array
    C_t = np.array(C_t)
    C_ref = np.array(C_ref)
    t = np.array(t)

    # Calculate integrals using trapezoidal rule
    integral_Ct = np.array([np.trapz(C_t[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Ct = np.insert(integral_Ct, 0, 0)  # Insert 0 at the beginning for alignment
    integral_Cref = np.array([np.trapz(C_ref[:i+1], t[:i+1]) for i in range(1, len(t))])
    integral_Cref = np.insert(integral_Cref, 0, 0)  # Insert 0 at the beginning for alignment

    # mask the data
    mask = (t >= t_trunc[0]) & (t <= t_trunc[1])
    t = t[mask]
    C_t = C_t[mask]
    C_ref = C_ref[mask]
    integral_Ct = integral_Ct[mask]
    integral_Cref = integral_Cref[mask]

    # Calculate normalized variables
    Logan_x = integral_Cref + C_ref / k2p
    Logan_y = C_t / integral_Ct

    # Calculate slopes between adjacent points
    slopes = np.gradient(Logan_y) / np.gradient(Logan_x)

    # Visualize MRTM2 plot with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Subplot (1, 1): MRTM2 equation normalized variables
    axes[0, 0].plot(Logan_x, Logan_y, label="$\\int_0^T C_R(t)dt + C_R(T) / k_2'$ vs $C_T(T) / \\int_0^T C_T(t)dt$", color="tab:blue")
    axes[0, 0].set_xlabel("$\\int_0^T C_R(t)dt + C_R(T) / k_2'$")
    axes[0, 0].set_ylabel("$C_T(T) / \\int_0^T C_T(t)dt$")
    axes[0, 0].legend()
    axes[0, 0].grid()

    # Subplot (2, 1): Slopes corresponding to subplot (1, 1)
    axes[1, 0].plot(Logan_x, slopes, label="Slopes", color="tab:red")
    axes[1, 0].set_xlabel("$\\int_0^T C_R(t)dt + C_R(T) / k_2'$")
    axes[1, 0].set_ylabel("Slopes")
    axes[1, 0].legend()
    axes[1, 0].grid()

    # Subplot (1, 2): Time vs normalized reference region
    axes[0, 1].plot(t, Logan_x, label="$t$ vs $\\int_0^T C_R(t)dt + C_R(T) / k_2'$", color="tab:green")
    axes[0, 1].set_xlabel("Time (t)")
    axes[0, 1].set_ylabel("$\\int_0^T C_R(t)dt + C_R(T) / k_2'$")
    axes[0, 1].legend()
    axes[0, 1].grid()

    # Subplot (2, 2): Time vs C_T(T) / ∫₀ᵀ C_T(t)dt
    axes[1, 1].plot(t, Logan_y, label="$t$ vs $C_T(T) / \\int_0^T C_T(t)dt$", color="tab:purple")
    axes[1, 1].set_xlabel("Time (t)")
    axes[1, 1].set_ylabel("$C_T(T) / \\int_0^T C_T(t)dt$")
    axes[1, 1].legend()
    axes[1, 1].grid()

    # Adjust layout
    fig.tight_layout()

    # Save the plot if save_dir is provided
    if save_dir is not None:
        file_path = f"{save_dir}/mrtm_k2p_linear_phase.svg"
        plt.savefig(file_path, format='svg')
        print(f"Plot saved to {file_path}")

    plt.show()
