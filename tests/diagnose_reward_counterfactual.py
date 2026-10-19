import numpy as np

def run_counterfactual():
    # Takeoff forward velocity analysis
    vy_vals = np.linspace(0.10, 0.45, 15)
    target_vy = 0.22
    sigma = 0.06
    w_vy = 3.0
    t_flight = 0.407  # typical flight time 0.407s (vz ~ 2.0 m/s -> 2 * vz / g ~ 0.408s)

    print("=== Takeoff Forward Velocity Reward Counterfactual ===")
    print(f"{'Vy (m/s)':<10} | {'Est Flight Dist (m)':<20} | {'Gaussian Rew':<14} {'dRew/dVy (1/(m/s))':<20} | {'Monotonic Sigmoid':<18} {'dSigmoid/dVy':<14}")
    print("-" * 90)

    for vy in vy_vals:
        dist_est = vy * t_flight
        # Gaussian
        diff = vy - target_vy
        gauss_val = w_vy * np.exp(-(diff**2) / (2.0 * sigma**2))
        gauss_grad = -w_vy * (diff / (sigma**2)) * np.exp(-(diff**2) / (2.0 * sigma**2))

        # Monotonic Sigmoid candidate (center 0.22, scale 0.05)
        z = (vy - 0.22) / 0.05
        sig_val = w_vy * (1.0 / (1.0 + np.exp(-z)))
        sig_grad = w_vy * (np.exp(-z) / ((1.0 + np.exp(-z))**2)) / 0.05

        print(f"{vy:<10.3f} | {dist_est:<20.3f} | {gauss_val:<14.4f} {gauss_grad:<20.4f} | {sig_val:<18.4f} {sig_grad:<14.4f}")

    print("\n=== Key Observations ===")
    print(f"1. At baseline vy = 0.22 m/s: Gaussian rew = {w_vy:.2f}, grad = 0.0.")
    vy_target_15cm = 0.15 / t_flight
    diff_15 = vy_target_15cm - target_vy
    rew_15 = w_vy * np.exp(-(diff_15**2) / (2.0 * sigma**2))
    grad_15 = -w_vy * (diff_15 / (sigma**2)) * np.exp(-(diff_15**2) / (2.0 * sigma**2))
    print(f"2. To achieve 15 cm jump distance, required vy = {vy_target_15cm:.3f} m/s.")
    print(f"   Current Gaussian rew at {vy_target_15cm:.3f} m/s = {rew_15:.4f} (drop of {((w_vy - rew_15)/w_vy)*100:.1f}%!).")
    print(f"   Reward gradient at this point = {grad_15:.2f} (HEAVILY OPPOSING forward acceleration!).")

if __name__ == "__main__":
    run_counterfactual()
