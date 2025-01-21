import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Example parameters corresponding (roughly) to Figure 3(a)
N          = 500      # "system size"
lambda_    = 5.0      # arrival rate λ
mu2        = 1.0      # service rate μ2
theta0     = 10.0     # parameter θ0
theta1     = 5.0      # parameter θ1 (if needed)
C          = 50.0     # constant in μ1(t)= C (q(t))^α1 (z0(t))^α2
alpha1     = 0.5
alpha2     = 0.5
Tmax       = 3.0      # run until time t = 3
dt_plot    = 0.01     # how finely to sample the solution for plotting

# Initial conditions (scaled so that q(0)+z0(0)+z2(0)=1 in the fluid model)
q0   = 1.0    # all “in queue” at t=0, for instance
z00  = 0.0
z20  = 0.0

# Stochastic initial conditions (in terms of counts)
Q0   = int(N*q0)
Z00  = int(N*z00)
Z20  = int(N*z20)


def fluid_rhs(t, x, params):
    """
    x = [q, z0, z2]
    params = (lambda_, theta0, C, alpha1, alpha2, mu2)
    """
    q, z0, z2 = x
    lam, th0, C_, a1, a2, mu2_ = params

    # Compute mu1(t) = C (q^alpha1)(z0^alpha2)
    mu1_t = C_ * (q ** a1) * (z0 ** a2)

    # Matching rate, e.g. mu1(t)* min(q,z0)
    match_rate = mu1_t * min(q, z0)

    dq = lam - th0 * q - match_rate
    dz0 = match_rate - mu2_ * z0
    dz2 = mu2_ * z0

    return [dq, dz0, dz2]


def solve_fluid_model(q0, z00, z20, params, Tmax, dt_plot=0.01):
    """
    Solves the fluid model from t=0 to t=Tmax.  Returns (t_grid, q_sol, z0_sol, z2_sol).
    """
    # initial condition vector
    x0 = [q0, z00, z20]

    # time points at which we want solution (for plotting)
    t_eval = np.arange(0, Tmax + dt_plot, dt_plot)

    sol = solve_ivp(fun=lambda t, x: fluid_rhs(t, x, params),
                    t_span=(0, Tmax), y0=x0, t_eval=t_eval)

    q_sol = sol.y[0, :]
    z0_sol = sol.y[1, :]
    z2_sol = sol.y[2, :]
    return sol.t, q_sol, z0_sol, z2_sol


def simulate_stochastic(N, lambda_, mu2, theta0, C, alpha1, alpha2,
                        Q0, Z00, Z20, Tmax, max_events=int(1e6)):
    """
    Discrete-event (Gillespie-style) simulation of the stochastic model up to time Tmax.
    Returns arrays of (t_points, Qvals, Z0vals, Z2vals).
    """
    t = 0.0
    Q = Q0
    Z0 = Z00
    Z2 = Z20

    # For storing sample path at discrete time points
    times = [t]
    Q_array = [Q]
    Z0_array = [Z0]
    Z2_array = [Z2]

    event_count = 0
    while t < Tmax and event_count < max_events:

        # ---- 1) Compute rates of each possible event  ----
        # arrivals (assuming rate = lambda_)
        rate_arrivals = lambda_

        # matching, e.g. mu1(t)* min(Q,Z0), where mu1(t) = C (q^alpha1)(z0^alpha2)
        # but now q = Q/N, z0 = Z0/N
        q_float = Q / float(N)
        z0_float = Z0 / float(N)
        mu1_t = C * (q_float ** alpha1) * (z0_float ** alpha2)
        rate_match = mu1_t * min(Q, Z0)

        # transitions from Z0 to Z2 at rate mu2 * Z0
        rate_z0_to_z2 = mu2 * Z0

        # total rate
        rate_total = rate_arrivals + rate_match + rate_z0_to_z2
        if rate_total <= 0:
            # no events can happen; break out
            break

        # ---- 2) Draw time to next event ----
        dt = np.random.exponential(1.0 / rate_total)
        t_next = t + dt

        # If overshooting Tmax, stop exactly at Tmax
        if t_next > Tmax:
            t_next = Tmax
            # We won't do another event; just record final state
            t = t_next
            times.append(t)
            Q_array.append(Q)
            Z0_array.append(Z0)
            Z2_array.append(Z2)
            break

        # Otherwise accept the event
        t = t_next
        event_count += 1

        # ---- 3) Decide which event type occurs ----
        u = np.random.rand() * rate_total
        if u < rate_arrivals:
            # arrival
            Q += 1
        elif u < rate_arrivals + rate_match:
            # matching from Q to Z0
            Q -= 1
            Z0 += 1
        else:
            # transition from Z0 to Z2
            Z0 -= 1
            Z2 += 1

        # record the state
        times.append(t)
        Q_array.append(Q)
        Z0_array.append(Z0)
        Z2_array.append(Z2)

    return np.array(times), np.array(Q_array), np.array(Z0_array), np.array(Z2_array)


def main():
    # --- 1) Set parameters (as above) ---
    N = 500
    lambda_ = 5.0
    mu2 = 1.0
    theta0 = 10.0
    C = 50.0
    alpha1 = 0.5
    alpha2 = 0.5
    Tmax = 3.0

    # fluid-model initial (scaled) conditions
    q0 = 1.0
    z00 = 0.0
    z20 = 0.0

    # stochastic initial (integer) conditions
    Q0 = int(N * q0)
    Z00 = int(N * z00)
    Z20 = int(N * z20)

    # --- 2) Solve fluid ODE ---
    params_fluid = (lambda_, theta0, C, alpha1, alpha2, mu2)
    t_f, q_f, z0_f, z2_f = solve_fluid_model(q0, z00, z20,
                                             params_fluid, Tmax)

    # --- 3) Run one sample-path of the stochastic system ---
    t_s, Q_s, Z0_s, Z2_s = simulate_stochastic(
        N, lambda_, mu2, theta0, C, alpha1, alpha2,
        Q0, Z00, Z20, Tmax
    )
    # Normalize by N
    q_s_norm = Q_s / N
    z0_s_norm = Z0_s / N
    z2_s_norm = Z2_s / N

    # --- 4) Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(t_f, q_f, 'r-', label="q(t) fluid")
    ax.plot(t_f, z0_f, 'b-', label="z0(t) fluid")
    ax.plot(t_f, z2_f, 'k-', label="z2(t) fluid")

    ax.plot(t_s, q_s_norm, 'r--', label="Q(t)/N stochastic")
    ax.plot(t_s, z0_s_norm, 'b--', label="Z0(t)/N stochastic")
    ax.plot(t_s, z2_s_norm, 'k--', label="Z2(t)/N stochastic")

    ax.set_xlim([0, Tmax])
    ax.set_ylim([0, 1])
    ax.set_xlabel("time t")
    ax.set_ylabel("Scaled populations")
    ax.legend()
    plt.title("Comparison of Fluid and Stochastic Models (N=500)")
    plt.show()


if __name__ == "__main__":
    main()
