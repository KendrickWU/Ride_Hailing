import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the RHS of the differential equations
def fluid_rhs(t, x, parameters):
    """
    Differential equations for the fluid model with two types of customers and corrected dynamics.

    Parameters:
    t: float
        Time variable.
    x: list [q1, q2, z0, z11, z12, z21, z22]
        State variables:
        - q1(t): normalized number of VIP passengers
        - q2(t): normalized number of regular passengers
        - z0(t): fraction of idle drivers
        - z11(t): fraction of drivers assigned to pick up VIP customers
        - z12(t): fraction of drivers completing VIP trips
        - z21(t): fraction of drivers assigned to pick up regular customers
        - z22(t): fraction of drivers completing regular trips
    parameters: tuple
        - lam1, lam2: arrival rates for VIP and regular customers
        - theta01, theta02: patience rates for VIP and regular customers
        - theta11, theta12: pick-up abandonment rates for VIP and regular customers
        - mu1_threshold1, mu1_threshold2: matching thresholds for VIP and regular customers
        - C1, C2, a1, a2: constants for mu1_t calculation
        - mu2: trip completion rate (same for both types)

    Returns:
    List of derivatives [dq1/dt, dq2/dt, dz0/dt, dz11/dt, dz12/dt, dz21/dt, dz22/dt].
    """
    q1, q2, z0, z11, z12, z21, z22 = x
    (lam1, lam2, theta01, theta02, theta11, theta12,
     mu1_threshold1, mu1_threshold2, C1, C2, a1, a2, mu2) = parameters

    # Ensure positive values to avoid invalid power operations
    q1, q2 = max(q1, 1e-8), max(q2, 1e-8)
    z0 = max(z0, 1e-8)

    # Initialize storage for match timing
    if not hasattr(fluid_rhs, "last_match_time"):
        fluid_rhs.last_match_time = [None, None]  # [VIP match time, Regular match time]
        fluid_rhs.last_matchs = [0, 0]  # [VIP matches, Regular matches]

    # Matching rates for VIP and regular customers
    mu1_t1 = C1 * (q1**a1) * (z0**a2)
    mu1_t2 = C2 * (q2**a1) * (z0**a2)

    # Matching policy for VIP customers
    if mu1_t1 >= mu1_threshold1:
        matchs1 = 0e-5
        while True:
            matchs1 += 1e-5
            new_mu1_t1 = (
                C1 * ((q1 - matchs1)**a1) * ((z0 - matchs1)**a2)
                if q1 > matchs1 and z0 > matchs1
                else 0
            )
            if new_mu1_t1 < mu1_threshold1:
                break
    else:
        matchs1 = 0

    # Matching policy for regular customers
    if mu1_t2 >= mu1_threshold2:
        matchs2 = 0e-5
        while True:
            matchs2 += 1e-5
            new_mu1_t2 = (
                C2 * ((q2 - matchs2)**a1) * ((z0 - matchs2)**a2)
                if q2 > matchs2 and z0 > matchs2
                else 0
            )
            if new_mu1_t2 < mu1_threshold2:
                break
    else:
        matchs2 = 0

    # Compute match rates (time between matches)
    if fluid_rhs.last_match_time[0] is not None and t > fluid_rhs.last_match_time[0]:
        match_duration1 = t - fluid_rhs.last_match_time[0]
        match_rate1 = matchs1 / match_duration1
    else:
        match_rate1 = 0

    if fluid_rhs.last_match_time[1] is not None and t > fluid_rhs.last_match_time[1]:
        match_duration2 = t - fluid_rhs.last_match_time[1]
        match_rate2 = matchs2 / match_duration2
    else:
        match_rate2 = 0

    # Update static state
    fluid_rhs.last_match_time = [t if matchs1 > 0 else fluid_rhs.last_match_time[0],
                                 t if matchs2 > 0 else fluid_rhs.last_match_time[1]]
    fluid_rhs.last_matchs = [matchs1, matchs2]

    # Driver constraint: z0 + z11 + z12 + z21 + z22 = 1
    z0 = max(1 - z11 - z12 - z21 - z22, 0)

    # Derivatives
    dq1_dt = lam1 - theta01 * q1 - match_rate1
    dq2_dt = lam2 - theta02 * q2 - match_rate2
    dz0_dt = mu2 * (z12 + z22) + z11 * theta11 + z21 * theta12 - match_rate1 - match_rate2
    dz11_dt = match_rate1 - (theta11 + mu1_threshold1) * z11
    dz12_dt = mu1_threshold1 * z11 - mu2 * z12
    dz21_dt = match_rate2 - (theta12 + mu1_threshold2) * z21
    dz22_dt = mu1_threshold2 * z21 - mu2 * z22

    return [dq1_dt, dq2_dt, dz0_dt, dz11_dt, dz12_dt, dz21_dt, dz22_dt]


# Parameters
params = (2, 5,       # lam1, lam2 (arrival rates for VIP and regular customers)
          12, 10,      # theta01, theta02 (patience rates)
          6, 5,       # theta11, theta12 (pickup abandonment rates)
          4, 8,      # mu1_threshold1, mu1_threshold2 (match thresholds)
          50, 50,     # C1, C2 (constants for matching rate calculation)
          0.5, 0.5,   # a1, a2 (exponents for matching rate calculation)
          1)          # mu2 (trip completion rate)

# Initial state: q1(0), q2(0), z0(0), z11(0), z12(0), z21(0), z22(0)
initial_state = [0, 0, 1, 0, 0, 0, 0]

# Time span for simulation
t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Solve the system
solution = solve_ivp(
    fun=lambda t, x: fluid_rhs(t, x, params),
    t_span=t_span,
    y0=initial_state,
    t_eval=t_eval
)

# Extract results
t_points = solution.t
q1_vals, q2_vals, z0_vals, z11_vals, z12_vals, z21_vals, z22_vals = solution.y

# Plot results
plt.figure(figsize=(12, 10))

# Plot customer queues
plt.subplot(3, 1, 1)
plt.plot(t_points, q1_vals, label="q1(t): VIP passengers", color="blue")
plt.plot(t_points, q2_vals, label="q2(t): Regular passengers", color="orange")
plt.title("Passenger Queues")
plt.xlabel("Time (t)")
plt.ylabel("Queue Length")
plt.legend()
plt.grid()

# Plot idle drivers
plt.subplot(3, 1, 2)
plt.plot(t_points, z0_vals, label="z0(t): Idle drivers", color="green")
plt.title("Idle Drivers")
plt.xlabel("Time (t)")
plt.ylabel("Fraction of Drivers")
plt.legend()
plt.grid()

# Plot assigned and busy drivers
plt.subplot(3, 1, 3)
plt.plot(t_points, z11_vals, label="z11(t): Assigned to VIP", color="purple")
plt.plot(t_points, z12_vals, label="z12(t): Busy with VIP", linestyle="--", color="purple")
plt.plot(t_points, z21_vals, label="z21(t): Assigned to Regular", color="brown")
plt.plot(t_points, z22_vals, label="z22(t): Busy with Regular", linestyle="--", color="brown")
plt.title("Assigned and Busy Drivers")
plt.xlabel("Time (t)")
plt.ylabel("Fraction of Drivers")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
