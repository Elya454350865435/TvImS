import numpy as np
from scipy.stats import t, chi2

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46,
    9.63, 4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26,
    -17.19, 9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14,
    -8.98, -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68
])

alpha = 0.20
n = len(data)
df = n - 1

mean = np.mean(data)
s2 = np.var(data, ddof=1)

t_critical = t.ppf(1 - alpha/2, df)
standard_error = np.sqrt(s2 / n)
a_lower = mean - t_critical * standard_error
a_upper = mean + t_critical * standard_error

chi2_lower_critical = chi2.ppf(1 - alpha/2, df)
chi2_upper_critical = chi2.ppf(alpha/2, df)
sigma2_lower = (df * s2) / chi2_lower_critical
sigma2_upper = (df * s2) / chi2_upper_critical

print(f"Доверительный интервал для параметра a")
print(f"  Нижняя граница: {a_lower:.6f}")
print(f"  Верхняя граница: {a_upper:.6f}")

print(f"\nДоверительный интервал для параметра σ²")
print(f"  Нижняя граница: {sigma2_lower:.6f}")
print(f"  Верхняя граница: {sigma2_upper:.6f}")