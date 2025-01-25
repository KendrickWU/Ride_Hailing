import numpy as np
import matplotlib.pyplot as plt

# Customizing the plot style for a high-end journal
plt.rcParams.update({
    'text.usetex': False,  # Disable LaTeX rendering
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'font.family': 'serif',
})


def simulate_ride_hailing(K, lambda_passenger, mu_1_threshold, mu_2_trip,
                          theta_0, theta_1, C, alpha1, alpha2, t_max=3.0, dt=0.01):
    C_normalized = C / (K ** (alpha1 + alpha2))
    time_steps = int(t_max / dt)
    time = np.linspace(0, t_max, time_steps)
    Q = np.zeros(time_steps, dtype=int)
    Z0 = np.zeros(time_steps, dtype=int)
    Z1 = np.zeros(time_steps, dtype=int)
    Z2 = np.zeros(time_steps, dtype=int)
    Z0[0] = K
    Q[0] = 0

    for t in range(1, time_steps):
        arrivals = np.random.poisson(lambda_passenger * dt)
        Q_current = Q[t - 1] + arrivals
        abandon_prob = 1 - np.exp(-theta_0 * dt)
        abandons = np.random.binomial(Q_current, abandon_prob)
        Q_current = max(Q_current - abandons, 0)
        Z0_prev = Z0[t - 1]
        mu_1_t = C_normalized * (Q_current ** alpha1) * (Z0_prev ** alpha2)

        matches = 0
        if mu_1_t > mu_1_threshold:
            max_possible = min(Q_current, Z0_prev)
            low, high = 0, max_possible
            m_candidate = 0
            while low <= high:
                mid = (low + high) // 2
                mu_new = C_normalized * ((Q_current - mid) ** alpha1) * ((Z0_prev - mid) ** alpha2)
                if mu_new <= mu_1_threshold:
                    m_candidate = mid
                    high = mid - 1
                else:
                    low = mid + 1
            matches = min(m_candidate, max_possible)

        Q_current = max(Q_current - matches, 0)
        Z0_current = Z0_prev - matches
        Z1_current = Z1[t - 1] + matches

        cancel_prob = 1 - np.exp(-theta_1 * dt)
        cancels = np.random.binomial(Z1_current, cancel_prob)
        Z1_current = max(Z1_current - cancels, 0)
        Z0_current += cancels

        pickup_prob = 1 - np.exp(-theta_0 * dt)
        pickups = np.random.binomial(Z1_current, pickup_prob)
        Z1_current = max(Z1_current - pickups, 0)
        Z2_current = Z2[t - 1] + pickups

        completion_prob = 1 - np.exp(-mu_2_trip * dt)
        completions = np.random.binomial(Z2_current, completion_prob)
        Z2_current = max(Z2_current - completions, 0)
        Z0_current += completions

        Q[t] = Q_current
        Z0[t] = Z0_current
        Z1[t] = Z1_current
        Z2[t] = Z2_current

    Q_scaled = Q / K
    Z0_scaled = Z0 / K
    Z1_scaled = Z1 / K
    Z2_scaled = Z2 / K

    return time, Q_scaled, Z0_scaled, Z1_scaled, Z2_scaled

# Parameters for simulation
K = 500
lambda_passenger = 5 * K
mu_1_threshold = 10
mu_2_trip = 1
theta_0 = 10
theta_1 = 5
C = 50
alpha1, alpha2 = 0.5, 0.5

time, Q_scaled, Z0_scaled, Z1_scaled, Z2_scaled = simulate_ride_hailing(
    K, lambda_passenger, mu_1_threshold, mu_2_trip,
    theta_0, theta_1, C, alpha1, alpha2
)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(time, Q_scaled, label=r'Waiting Passengers $Q(t)/N$', color='red', linestyle=':', linewidth=1)
plt.plot(time, Z0_scaled, label=r'Idle Drivers $Z_0(t)/N$', color='blue', linestyle=':', linewidth=1)
plt.plot(time, Z2_scaled, label=r'Busy Drivers $Z_2(t)/N$', color='black', linestyle=':', linewidth=1)

# Title and labels
plt.title(r'Stochastic Ride-Hailing Process (Scaled by $N$)', fontsize=14)
plt.xlabel(r'Time $t$', fontsize=12)
plt.ylabel(r'Scaled Process', fontsize=12)

# Enhancing legend and grid
plt.legend(loc='upper right', frameon=False)
#plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')

# Show the plot
plt.show()
