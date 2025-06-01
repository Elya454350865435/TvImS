import numpy as np

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46, 9.63,
    4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26, -17.19,
    9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14, -8.98,
    -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68
])

n = len(data)

mmp_mean = np.mean(data)
mmp_var = np.var(data, ddof=0)
moments_mean = np.mean(data)
moments_var = np.var(data, ddof=1)

bias_mmp_var = -moments_var / n

print(f"ММП: a = {mmp_mean:.4f}, σ² = {mmp_var:.4f}")
print(f"Метод моментов: a = {moments_mean:.4f}, σ² = {moments_var:.4f}")
print(f"Смещения ММП: a = 0.0000, σ² = {bias_mmp_var:.4f}")
print(f"Смещения метода моментов: a = 0.0000, σ² = 0.0000")