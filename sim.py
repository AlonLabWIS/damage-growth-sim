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
    t_span = (0, 30)
    t_eval = np.linspace(*t_span, 100)
    solution = solve_ivp(
        bacterial_growth, t_span, [y0, x0], t_eval=t_eval, args=(k, T, alpha, c, r)
    )

    # Plot Results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(solution.t, solution.y[0], label="Damage (y)", color="red")
    ax.plot(solution.t, solution.y[1], label="Bacterial Density (x)", color="blue")
    ax.axhline(T, linestyle="--", color="gray", label="Threshold T")
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(f"Simulation with k={k}, T={T}, α={alpha}, c={c}, r={r}")

    st.pyplot(fig)

if __name__ == "__main__":
    # Streamlit App
    st.title("Bacterial Growth Simulation")
    st.sidebar.header("Adjust Parameters")

    k = st.sidebar.slider("Carrying Capacity (k)", 0.1, 1.1, 1.0, 0.05)
    T = st.sidebar.slider("Threshold (T)", 0.05, 1.0, 0.5, 0.05)
    alpha = st.sidebar.slider("Damage Rate (α)", 0.01, 10.0, 0.1, 0.01)
    c = st.sidebar.slider("Concentration (c)", 0.0, 10.0, 0.2, 0.1)
    x0 = st.sidebar.slider("Initial Bacterial Density (x₀)", 0.01, 1.0, 0.1, 0.01)
    y0 = st.sidebar.slider("Initial Damage (y₀)", 0.0, 1.0, 0.0, 0.01)
    r = st.sidebar.slider("Growth Rate (r)", 0.1, 5.0, 1.0, 0.1)

    simulate(k, T, alpha, c, x0, y0, r)
