import numpy as np
import matplotlib.pyplot as plt


def simulate_two_types(
        K,
        lambda_1, lambda_2,  # VIP/regular arrival rates
        mu1_threshold_1, mu1_threshold_2,  # Matching thresholds
        theta01, theta02,  # Patience for matching (VIP/regular)
        theta11, theta12,  # Patience for pickup (VIP/regular)
        C1, C2,  # Spatial constants for VIP/regular
        alpha1_1, alpha2_1,  # Spatial exponents for VIP
        alpha1_2, alpha2_2,  # Spatial exponents for regular
        mu2_trip_1, mu2_trip_2,  # Trip completion rates (VIP/regular)
        t_max=5.0,
        dt=0.01
):
    # Normalize spatial constants by K^(alpha1 + alpha2)
    C1_normalized = C1 / (K ** (alpha1_1 + alpha2_1))
    C2_normalized = C2 / (K ** (alpha1_2 + alpha2_2))

    # Time settings
    time_steps = int(t_max / dt)
    time = np.linspace(0, t_max, time_steps)

    # State variables (integer counts)
    Q_vip = np.zeros(time_steps, dtype=int)  # VIP waiting passengers
    Q_reg = np.zeros(time_steps, dtype=int)  # Regular waiting passengers
    Z0 = np.zeros(time_steps, dtype=int)  # Idle drivers
    z11 = np.zeros(time_steps, dtype=int)  # VIP assigned drivers
    z12 = np.zeros(time_steps, dtype=int)  # VIP busy drivers
    z21 = np.zeros(time_steps, dtype=int)  # Regular assigned drivers
    z22 = np.zeros(time_steps, dtype=int)  # Regular busy drivers

    # Initialize
    Z0[0] = K  # All drivers start idle

    # Simulation loop
    for t in range(1, time_steps):
        # --- Passenger arrivals ---
        # VIP arrivals
        arrivals_vip = np.random.poisson(lambda_1 * dt)
        Q_vip_current = Q_vip[t - 1] + arrivals_vip

        # Regular arrivals
        arrivals_reg = np.random.poisson(lambda_2 * dt)
        Q_reg_current = Q_reg[t - 1] + arrivals_reg

        # --- Passenger abandonments ---
        # VIP abandonments (matching)
        abandon_prob_vip = 1 - np.exp(-theta01 * dt)
        abandons_vip = np.random.binomial(Q_vip_current, abandon_prob_vip)
        Q_vip_current = max(Q_vip_current - abandons_vip, 0)

        # Regular abandonments (matching)
        abandon_prob_reg = 1 - np.exp(-theta02 * dt)
        abandons_reg = np.random.binomial(Q_reg_current, abandon_prob_reg)
        Q_reg_current = max(Q_reg_current - abandons_reg, 0)

        # --- Matching process (VIP first) ---
        Z0_prev = Z0[t - 1]

        # 1. VIP matching
        mu1_vip = C1_normalized * (Q_vip_current ** alpha1_1) * (Z0_prev ** alpha2_1)
        matches_vip = 0
        if mu1_vip > mu1_threshold_1:
            max_possible = min(Q_vip_current, Z0_prev)
            low, high = 0, max_possible
            m_candidate = 0
            while low <= high:
                mid = (low + high) // 2
                mu_new = C1_normalized * ((Q_vip_current - mid) ** alpha1_1) * ((Z0_prev - mid) ** alpha2_1)
                if mu_new <= mu1_threshold_1:
                    m_candidate = mid
                    high = mid - 1
                else:
                    low = mid + 1
            matches_vip = min(m_candidate, max_possible)

        # Update VIP queue and drivers
        Q_vip_current = max(Q_vip_current - matches_vip, 0)
        Z0_current = max(Z0_prev - matches_vip,0)
        z11_current = z11[t - 1] + matches_vip

        # 2. Regular matching (uses remaining Z0)
        mu1_reg = C2_normalized * (Q_reg_current ** alpha1_2) * (Z0_current ** alpha2_2)
        matches_reg = 0
        if mu1_reg > mu1_threshold_2:
            max_possible = min(Q_reg_current, Z0_current)
            low, high = 0, max_possible
            m_candidate = 0
            while low <= high:
                mid = (low + high) // 2
                mu_new = C2_normalized * ((Q_reg_current - mid) ** alpha1_2) * ((Z0_current - mid) ** alpha2_2)
                if mu_new <= mu1_threshold_2:
                    m_candidate = mid
                    high = mid - 1
                else:
                    low = mid + 1
            matches_reg = min(m_candidate, max_possible)

        # Update regular queue and drivers
        Q_reg_current = max(Q_reg_current - matches_reg, 0)
        Z0_current = max(Z0_current - matches_reg, 0)
        z21_current = z21[t - 1] + matches_reg

        # --- Cancellations (assigned drivers) ---
        # VIP cancellations
        cancel_prob_vip = 1 - np.exp(-theta11 * dt)
        cancels_vip = np.random.binomial(z11_current, cancel_prob_vip)
        z11_current = max(z11_current - cancels_vip, 0)
        Z0_current += cancels_vip  # Drivers return to idle

        # Regular cancellations
        cancel_prob_reg = 1 - np.exp(-theta12 * dt)
        cancels_reg = np.random.binomial(z21_current, cancel_prob_reg)
        z21_current = max(z21_current - cancels_reg, 0)
        Z0_current += cancels_reg  # Drivers return to idle

        # --- Successful pickups (assigned -> busy) ---
        # VIP pickups
        pickup_prob_vip = 1 - np.exp(-mu1_vip * dt)
        pickups_vip = np.random.binomial(z11_current, pickup_prob_vip)
        z11_current = max(z11_current - pickups_vip, 0)
        z12_current = z12[t - 1] + pickups_vip

        # Regular pickups
        pickup_prob_reg = 1 - np.exp(-mu1_reg * dt)
        pickups_reg = np.random.binomial(z21_current, pickup_prob_reg)
        z21_current = max(z21_current - pickups_reg, 0)
        z22_current = z22[t - 1] + pickups_reg

        # --- Trip completions (busy -> idle) ---
        # VIP trip completions
        completion_prob_vip = 1 - np.exp(-mu2_trip_1 * dt)
        completions_vip = np.random.binomial(z12_current, completion_prob_vip)
        z12_current = max(z12_current - completions_vip, 0)
        Z0_current += completions_vip

        # Regular trip completions
        completion_prob_reg = 1 - np.exp(-mu2_trip_2 * dt)
        completions_reg = np.random.binomial(z22_current, completion_prob_reg)
        z22_current = max(z22_current - completions_reg, 0)
        Z0_current += completions_reg

        # Ensure driver conservation
        assert Z0_current + z11_current + z12_current + z21_current + z22_current == K, "Driver count mismatch!"

        # Update state arrays
        Q_vip[t] = Q_vip_current
        Q_reg[t] = Q_reg_current
        Z0[t] = Z0_current
        z11[t] = z11_current
        z12[t] = z12_current
        z21[t] = z21_current
        z22[t] = z22_current

    # Scale by K for plotting
    Q_vip_scaled = Q_vip / K
    Q_reg_scaled = Q_reg / K
    Z0_scaled = Z0 / K
    z12_scaled = z12 / K
    z22_scaled = z22 / K

    return time, Q_vip_scaled, Q_reg_scaled, Z0_scaled, z12_scaled, z22_scaled


# Example parameters (adjust based on your case)
K = 10000
lambda_1 = 2 * K  # VIP arrival rate (Nλ₁)
lambda_2 = 5 * K  # Regular arrival rate (Nλ₂)
mu1_threshold_1, mu1_threshold_2 = 4, 8  # Matching thresholds
theta01, theta02 = 12, 10  # Patience for matching
theta11, theta12 = 6, 5  # Patience for pickup
C1, C2 = 50, 50  # Spatial constants
alpha1_1, alpha2_1 = 0.5, 0.5  # VIP spatial exponents
alpha1_2, alpha2_2 = 0.5, 0.5  # Regular spatial exponents
mu2_trip_1, mu2_trip_2 = 1, 1  # Trip completion rates

# Run simulation
time, Q_vip, Q_reg, Z0, z12, z22 = simulate_two_types(
    K, lambda_1, lambda_2, mu1_threshold_1, mu1_threshold_2,
    theta01, theta02, theta11, theta12, C1, C2,
    alpha1_1, alpha2_1, alpha1_2, alpha2_2,
    mu2_trip_1, mu2_trip_2
)


# Plot results with improved aesthetics
plt.figure(figsize=(10, 6))
plt.plot(time, Q_vip, label=r'VIP $Q(t)/N$', color='red', linestyle='-', linewidth=2)
plt.plot(time, Q_reg, label=r'Regular $Q(t)/N$', color='orange', linestyle='--', linewidth=2)
plt.plot(time, Z0, label=r'Idle Drivers $Z_0(t)/N$', color='blue', linestyle='-', linewidth=2)
plt.plot(time, z12, label=r'VIP Busy Drivers $Z_{12}(t)/N$', color='purple', linestyle='-.', linewidth=2)
plt.plot(time, z22, label=r'Regular Busy Drivers $Z_{22}(t)/N$', color='green', linestyle=':', linewidth=2)

# Title and labels with larger fonts for readability
plt.title('Two-Type Ride-Hailing Stochastic Process (Scaled by $N$)', fontsize=16)
plt.xlabel('Time $t$', fontsize=14)
plt.ylabel('Scaled Variables', fontsize=14)

# Enhanced legend and layout
plt.legend(loc='upper right', fontsize=12, frameon=False)

# Adjust tick parameters for a cleaner look
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Remove grid and use a cleaner background
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.show()