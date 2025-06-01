import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46,
    9.63, 4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26,
    -17.19, 9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14,
    -8.98, -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68
])

alpha = 0.20
h = 4.0

a_hat = np.mean(data)
sigma_hat = np.std(data, ddof=0)

bins = np.arange(np.floor(data.min()), np.ceil(data.max()) + h, h)
observed, bin_edges = np.histogram(data, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], observed, width=h, align='edge', alpha=0.7,
        edgecolor='black', label='Гистограмма')
plt.plot(bin_centers, observed, 'ro-', label='Полигон частот')
x_plot = np.linspace(-40, 30, 1000)
plt.plot(x_plot, len(data)*h*norm.pdf(x_plot, loc=a_hat, scale=sigma_hat),
         'g-', label=f'Теоретическая N({a_hat:.2f}, {sigma_hat**2:.2f})')
plt.title(f'Гистограмма и нормальное распределение (сложная гипотеза)')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chi2_complex_hist.png")
plt.show()

grouped_bins = [-25, -17, -9, -1, 7, 15, 23]
grouped_observed = [
    sum(observed[:3]),
    sum(observed[3:5]),
    sum(observed[5:7]),
    sum(observed[7:9]),
    sum(observed[9:11]),
    observed[11]
]

n = len(data)
expected = []
for i in range(len(grouped_bins) - 1):
    left = grouped_bins[i]
    right = grouped_bins[i + 1]
    p = norm.cdf((right - a_hat) / sigma_hat) - norm.cdf((left - a_hat) / sigma_hat)
    expected.append(p * n)

chi2_stat = np.sum((np.array(grouped_observed) - np.array(expected))**2 / np.array(expected))
k = len(grouped_observed)
r = 2
df = k - 1 - r
p_value = 1 - chi2.cdf(chi2_stat, df)
critical_value = chi2.ppf(1 - alpha, df)

print("\nОбъединённые интервалы и частоты:")
for i in range(len(grouped_bins) - 1):
    print(f"[{grouped_bins[i]:.0f}, {grouped_bins[i+1]:.0f}): "
          f"O = {grouped_observed[i]}, E = {expected[i]:.2f}")

print(f"\nСтатистика χ² = {chi2_stat:.4f}")
print(f"Критическое значение χ² при α = {alpha}: {critical_value:.4f}")
print(f"p-value = {p_value:.4f}")

if p_value < alpha:
    print(f"Гипотеза H₀ отвергается на уровне значимости α = {alpha}")
else:
    print(f"Нет оснований отвергнуть гипотезу H₀ при α = {alpha}")

print(f"Наибольший уровень значимости, при котором H₀ ещё принимается: {p_value:.4f}")
