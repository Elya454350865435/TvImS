import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46,
    9.63, 4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26,
    -17.19, 9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14,
    -8.98, -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68
])

n = len(data)
alpha = 0.20

a0, sigma0 = -36.0, 10.0

a1, sigma1 = -3.0, 10.0

def log_likelihood_ratio(x):
    term = ((x - a1)**2 - (x - a0)**2) / (2 * sigma0**2)
    return np.sum(term)

T_stat = log_likelihood_ratio(data)

simulations = 100_000
T_H0 = np.array([
    log_likelihood_ratio(np.random.normal(loc=a0, scale=sigma0, size=n))
    for _ in range(simulations)
])

T_crit = np.quantile(T_H0, alpha)

T_H1 = np.array([
    log_likelihood_ratio(np.random.normal(loc=a1, scale=sigma1, size=n))
    for _ in range(simulations)
])

power = np.mean(T_H1 < T_crit)

print(f"Статистика T = {T_stat:.4f}")
print(f"Критическое значение T_crit = {T_crit:.4f}")
print(f"Мощность критерия γ = {power:.4f}")
if T_stat < T_crit:
    print("H₀ отвергается в пользу H₁")
else:
    print("Нет оснований отвергнуть H₀")

plt.figure(figsize=(10, 5))
plt.hist(T_H0, bins=100, alpha=0.6, label="H₀: N(-36, 100)", color='skyblue', density=True)
plt.hist(T_H1, bins=100, alpha=0.6, label="H₁: N(-3, 100)", color='salmon', density=True)
plt.axvline(T_crit, color='red', linestyle='--', label='Критическая граница T')
plt.axvline(T_stat, color='black', linestyle='-', label='Статистика T (набл.)')
plt.title("Распределения логарифма отношения правдоподобия")
plt.xlabel("T")
plt.ylabel("Плотность")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("task8_fixed_corrected.png")
plt.show()
