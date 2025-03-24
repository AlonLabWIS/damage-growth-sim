import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st

# System of ODEs
def bacterial_growth(t, state, k, T, alpha, c, r):
    y, x = state
    dy_dt = r * (alpha * c - y)
    dx_dt = r * x * (1 - x / k) * (y < T)
    return [dy_dt, dx_dt]

# Main Simulation Function
def simulate(k, T, alpha, c, x0, y0, r):
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 100)
    solution = solve_ivp(
        bacterial_growth, t_span, [y0, x0], t_eval=t_eval, args=(k, T, alpha, c, r)
    )

    return solution.t, solution.y[0], solution.y[1]  # Return time, damage, and bacterial density

if __name__ == "__main__":

    # Set rcParams for larger annotations in plots
    plt.rcParams.update({
        'font.size': 14,        # Increase font size
        'axes.titlesize': 16,   # Title font size
        'axes.labelsize': 14,   # Axis label font size
        'xtick.labelsize': 12,  # X-axis tick label size
        'ytick.labelsize': 12,  # Y-axis tick label size
        'legend.fontsize': 12   # Legend font size
    })

    params_1 = {}
    params_2 = {}

    # Streamlit App
    st.title("Bacterial Growth Simulation")


    # st.title("Parameters")

    # Sidebar Inputs
    param_descriptions = {
        "r": "Growth Rate",
        "k": "Carrying Capacity",
        "T": "Damage Threshold",
        "alpha": "Coefficient of Conversion (Conc → Damage)",
        "c": "Concentration",
        "x0": "Initial Bacteria",
        "y0": "Initial Damage"
    }

    st.sidebar.header("Adjust Parameters")
    free_param = st.sidebar.selectbox(
        "Choose a parameter to vary:", list(param_descriptions.keys()), format_func=lambda x: f"{x} ({param_descriptions[x]})"
    )

    #(label, min_value, max_value, default_value, step)
    fixed_values = {}
    # Define parameter ranges in a dictionary
    parameter_ranges = {
        "k": (0.1, 1.1, 1.0, 0.05),
        "T": (0.05, 1.0, 0.5, 0.05),
        "alpha": (0.01, 5.0, 2.5, 0.02),
        "c": (0.0, 2.0, 0.2, 0.05),
        "x0": (0.01, 1.0, 0.1, 0.01),
        "y0": (0.0, 1.0, 0.0, 0.01),
        "r": (0.1, 5.0, 1.0, 0.1),
    }

    # Sidebar sliders for fixed parameters
    for param, (min_val, max_val, default, step) in parameter_ranges.items():
        if param != free_param:  # Exclude the free parameter
            fixed_values[param] = st.sidebar.slider(
                f"{param} ({param_descriptions[param]})", min_val, max_val, default, step
            )

    # Two values for the selected "free" parameter
    st.sidebar.markdown(f"**{free_param} ({param_descriptions[free_param]}) - Upper figure**")
    free_param_value_1 = st.sidebar.slider(
        free_param, *parameter_ranges[free_param], key="free_param_1"
    )

    st.sidebar.markdown(f"**{free_param} ({param_descriptions[free_param]}) - Lower figure**")
    free_param_value_2 = st.sidebar.slider(
        free_param, *parameter_ranges[free_param], key="free_param_2"
    )



    # Collect parameters
    params_1 = fixed_values.copy()
    params_2 = fixed_values.copy()
    params_1[free_param] = free_param_value_1
    params_2[free_param] = free_param_value_2

    # Run simulations
    t1, y1, x1 = simulate(**params_1)
    t2, y2, x2 = simulate(**params_2)

    # y_max = max(max(y1), max(x1), max(y2), max(x2))
    


    with st.container(border=True):
        st.subheader("Model equations", divider="grey")
        st.latex(r"\frac{dy}{dt} = r (\alpha c - y)")
        st.latex(r"\frac{dx}{dt} = r x \left(1 - \frac{x}{k} \right) \theta(y < T)")
        st.latex(r"\theta_{T}(y) = \begin{cases} 1, & y \le T \\ 0, & y > T \end{cases}")


    with st.container(border=True):

        fig, axes = plt.subplots(2, 1, figsize=(10, 15), gridspec_kw={'hspace': 0.8}, sharex=True)  # Adds vertical space

    
        # First subplot with dual y-axes
        ax1 = axes[0]
        ax2 = ax1.twinx()  # Create second y-axis

        ax1.plot(t1, x1, label="Bacterial Density (x)", color="blue", linewidth=5)
        ax2.plot(t1, y1, label="Damage (y)", color="red", linewidth=5)
        ax2.axhline(params_1["T"], linestyle="--", color="gray", label="Threshold T")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=12)



        ax1.set_xlabel("Time")
        ax1.set_ylabel("Bacterial Density (x)", color="blue")
        ax2.set_ylabel("Damage (y)", color="red")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax1.set_title(f"Simulation with {free_param}={free_param_value_1}")

        # ax1.legend(loc="center left", bbox_to_anchor=(1, 0.3))
        # ax2.legend(loc="center right", bbox_to_anchor=(1.15, 0.6))

        # Second subplot with dual y-axes
        ax3 = axes[1]
        ax4 = ax3.twinx()  # Create second y-axis

        ax3.plot(t2, x2, label="Bacterial Density (x)", color="blue", linewidth=5)
        ax4.plot(t2, y2, label="Damage (y)", color="red", linewidth=5)
        ax4.axhline(params_2["T"], linestyle="--", color="gray", label="Threshold T")

        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()

        ax3.legend(lines3 + lines4, labels3 + labels4, loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=12)



        ax3.set_xlabel("Time")
        ax3.set_ylabel("Bacterial Density (x)", color="blue")
        ax4.set_ylabel("Damage (y)", color="red")

        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)

        ax3.set_title(f"Simulation with {free_param}={free_param_value_2}")

        # Determine the max values for both axes
        max_x = max(max(x1), max(x2))  # Max value for bacterial density
        max_y = max(max(y1), max(y2))  # Max value for damage

        # Set the same y-axis limits for both subplots
        ax1.set_ylim(0, max_x * 1.1)  # Left y-axis (bacterial density)
        ax3.set_ylim(0, max_x * 1.1)  # Left y-axis for second plot

        ax2.set_ylim(0, max_y * 1.1)  # Right y-axis (damage)
        ax4.set_ylim(0, max_y * 1.1)  # Right y-axis for second plot

        st.pyplot(fig)

