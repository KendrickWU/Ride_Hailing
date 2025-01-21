import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 500  # Total number of drivers
lambda_passenger = 5  # Arrival rate of passengers
mu_1 = 10  # Matching rate
mu_2 = 1  # Trip completion rate
theta_0 = 10  # Patience for matching
theta_1 = 5  # Patience for pickup
C = 0.5  # Normalization constant for spatial model
alpha_1 = 0.5
alpha_2 = 0.5
mu_1_threshold = 5  # Matching threshold

# Normalizing parameters
lambda_passenger /= K
mu_1 /= K
mu_2 /= K
C /= K

# Time settings
t_max = 3.0
dt = 0.01
time_steps = int(t_max / dt)
time = np.linspace(0, t_max, time_steps)

# State variables
Q = np.zeros(time_steps)  # Waiting passengers
Z_0 = np.zeros(time_steps)  # Idle drivers
Z_1 = np.zeros(time_steps)  # Assigned drivers
Z_2 = np.zeros(time_steps)  # Busy drivers

# Process counters
R_0 = np.zeros(time_steps)  # Abandonments
R_1 = np.zeros(time_steps)  # Cancellations
D_1 = np.zeros(time_steps)  # Successful pickups
D_2 = np.zeros(time_steps)  # Completed trips
M = np.zeros(time_steps)  # Matchings

# Initial conditions
Z_0[0] = K
Q[0] = 0

# Simulation loop
for t in range(1, time_steps):
    # Arrival of passengers
    arrivals = np.random.poisson(lambda_passenger * dt)
    Q[t] = Q[t - 1] + arrivals

    # Abandonments
    abandon_probs = 1 - np.exp(-theta_0 * dt)
    abandons = np.random.binomial(Q[t - 1].astype(int), abandon_probs)
    R_0[t] = R_0[t - 1] + abandons
    Q[t] -= abandons

    # Matching process
    mu_1_t = C * (Q[t] ** alpha_1) * (Z_0[t - 1] ** alpha_2)
    if mu_1_t > mu_1_threshold:
        matches = min(Q[t], Z_0[t - 1])
    else:
        matches = 0
    M[t] = M[t - 1] + matches
    Q[t] -= matches
    Z_0[t] = Z_0[t - 1] - matches
    Z_1[t] = Z_1[t - 1] + matches

    # Cancellations
    cancel_probs = 1 - np.exp(-theta_1 * dt)
    cancels = np.random.binomial(Z_1[t - 1].astype(int), cancel_probs)
    R_1[t] = R_1[t - 1] + cancels
    Z_1[t] -= cancels

    # Successful pickups
    pickups = np.random.binomial(Z_1[t - 1].astype(int), mu_1 * dt)
    D_1[t] = D_1[t - 1] + pickups
    Z_1[t] -= pickups
    Z_2[t] += pickups

    # Completed trips
    completions = np.random.binomial(Z_2[t - 1].astype(int), mu_2 * dt)
    D_2[t] = D_2[t - 1] + completions
    Z_2[t] -= completions
    Z_0[t] += completions

# Normalizing the results for plotting
Q_scaled = Q / K
Z_0_scaled = Z_0 / K
Z_1_scaled = Z_1 / K
Z_2_scaled = Z_2 / K

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(time, Q_scaled, label='Q(t)/N', linestyle='--', color='red')
plt.plot(time, Z_0_scaled, label='Z0(t)/N', linestyle='--', color='blue')
plt.plot(time, Z_2_scaled, label='Z2(t)/N', linestyle='--', color='black')
plt.plot(time, Z_1_scaled, label='Z1(t)/N', linestyle='--', color='green')
plt.xlabel('Time')
plt.ylabel('Scaled Variables')
plt.legend()
plt.title('Scaled Stochastic Process for Ride-Hailing')
plt.grid()
plt.show()
