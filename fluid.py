import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the RHS of the differential equations
def fluid_rhs(t, x, parameters):
    """
    Differential equations for the fluid model with corrected matching policy.

    Parameters:
    t: float
        Time variable (unused but required by solve_ivp).
    x: list [q(t), z0(t), z1(t)]
        State variables:
        - q(t): normalized number of passengers
        - z0(t): fraction of idle drivers
        - z1(t): fraction of assigned drivers
    parameters: tuple
        - lambda_: passenger arrival rate
        - theta0: passenger abandonment rate
        - mu2: trip completion rate
        - theta1: trip pickup abandonment rate
        - mu1_threshold: pickup rate threshold
        - C_, a1, a2: constants for mu1_t calculation

    Returns:
    List of derivatives [dq/dt, dz0/dt, dz1/dt].
    """
    q, z0, z1 = x
    lam, theta0, mu2, theta1, mu1_threshold, C_, a1, a2 = parameters

    # Initialize storage for match timing
    static_state = hasattr(fluid_rhs, "last_match_time")
    if not static_state:
        fluid_rhs.last_match_time = None  # Last match time
        fluid_rhs.last_matchs = None      # Last number of matches

    # Ensure q and z0 are positive to avoid invalid power operations
    q = max(q, 1e-8)
    z0 = max(z0, 1e-8)

    # Compute mu1(t) = C * (q^a1) * (z0^a2)
    mu1_t = C_ * (q**a1) * (z0**a2)

    # Matching policy
    if mu1_t >= mu1_threshold:
        matchs = 0e-5
        while True:
            matchs += 1e-5  # Increment matches
            new_mu1_t = (
                C_ * ((q - matchs)**a1) * ((z0 - matchs)**a2)
                if q > matchs and z0 > matchs
                else 0
            )
            if new_mu1_t < mu1_threshold:
                break

        # Compute match_duration (time between two matches)
        if fluid_rhs.last_match_time is not None and t > fluid_rhs.last_match_time:
            match_duration = t - fluid_rhs.last_match_time
            match_rate = matchs / match_duration
        else:
            match_duration = float("inf")  # Avoid divide-by-zero
            match_rate = 0  # Default match rate for the first match

        # Update static state
        fluid_rhs.last_match_time = t
        fluid_rhs.last_matchs = matchs
    else:
        match_rate = 0

    # Driver constraint: z0 + z1 + z2 = 1
    z2 = max(1 - z0 - z1, 0)  # Ensure non-negative z2

    # Derivatives
    dq_dt = lam - theta0 * q - match_rate
    dz0_dt = z1 * theta1 + z2 * mu2 - match_rate
    dz1_dt = - (theta1 + mu1_threshold) * z1 + match_rate

    return [dq_dt, dz0_dt, dz1_dt]


# Common parameters
theta0_common = 10
theta1_common = 5
mu2_common = 1
n_drivers = 500  # Number of drivers (normalized)

# Case (a) parameters
params_a = (5, theta0_common, mu2_common, theta1_common, 10, 50, 0.5, 0.5)

# Case (b) parameters
params_b = (2, theta0_common, mu2_common, theta1_common, 5, 30, 0.3, 0.6)

# Initial state: q(0), z0(0), z1(0)
initial_state = [0, 1, 0]

# Time span for simulation
t_span = (0, 3)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve for Case (a)
solution_a = solve_ivp(
    fun=lambda t, x: fluid_rhs(t, x, params_a),
    t_span=t_span,
    y0=initial_state,
    t_eval=t_eval
)

# Solve for Case (b)
solution_b = solve_ivp(
    fun=lambda t, x: fluid_rhs(t, x, params_b),
    t_span=t_span,
    y0=initial_state,
    t_eval=t_eval
)

# Extract results for Case (a)
t_points_a = solution_a.t
q_vals_a, z0_vals_a, z1_vals_a = solution_a.y

# Extract results for Case (b)
t_points_b = solution_b.t
q_vals_b, z0_vals_b, z1_vals_b = solution_b.y

# Calculate z2(t) for both cases
z2_vals_a = 1 - z0_vals_a - z1_vals_a
z2_vals_b = 1 - z0_vals_b - z1_vals_b

# Plot results for both cases
plt.figure(figsize=(12, 8))

# Case (a)
plt.subplot(2, 1, 1)
plt.plot(t_points_a, q_vals_a, label="q(t): Passengers waiting", color="blue")
plt.plot(t_points_a, z0_vals_a, label="z0(t): Idle drivers", color="green")
plt.plot(t_points_a, z2_vals_a, label="z2(t): Drivers completing trips", color="red")
plt.title("Case (a): lambda=5, mu1=10, C=50, alpha1=alpha2=0.5")
plt.xlabel("Time (t)")
plt.ylabel("State Variables")
plt.legend()
plt.grid()

# Case (b)
plt.subplot(2, 1, 2)
plt.plot(t_points_b, q_vals_b, label="q(t): Passengers waiting", color="blue")
plt.plot(t_points_b, z0_vals_b, label="z0(t): Idle drivers", color="green")
plt.plot(t_points_b, z2_vals_b, label="z2(t): Drivers completing trips", color="red")
plt.title("Case (b): lambda=2, mu1=5, C=30, alpha1=0.3, alpha2=0.6")
plt.xlabel("Time (t)")
plt.ylabel("State Variables")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
