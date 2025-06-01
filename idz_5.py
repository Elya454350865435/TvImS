import numpy as np
import matplotlib.pyplot as plt
import math

erf_vectorized = np.vectorize(math.erf)

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46,
    9.63, 4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26,
    -17.19, 9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14,
    -8.98, -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68

])

alpha = 0.20
a0 = -36.00
sigma0 = 10.00

def ecdf(x, data_sorted):
    return np.searchsorted(data_sorted, x, side='right') / len(data_sorted)

def normal_cdf(x, a, sigma):
    z = (x - a) / (sigma * np.sqrt(2))
    return 0.5 * (1 + erf_vectorized(z))

data_sorted = np.sort(data)
n = len(data_sorted)
x = np.linspace(data_sorted[0] - 1, data_sorted[-1] + 1, 1000)

Fn = ecdf(x, data_sorted)
F0 = normal_cdf(x, a0, sigma0)

Dn = np.max(np.abs(Fn - F0))

L = Dn * np.sqrt(n)
p_value = 2 * np.exp(-2 * L**2)

critical_value = 1.07 / np.sqrt(n)
reject_h0 = Dn > critical_value

plt.figure(figsize=(10, 6))
plt.step(x, Fn, 'b-', label='Эмпирическая Fn(x)', linewidth=2)
plt.plot(x, F0, 'r-', label='Теоретическая F0(x)', linewidth=2)
plt.fill_between(x, F0 - critical_value, F0 + critical_value, color='gray', alpha=0.3, label='Доверительная область')
plt.title(f'Критерий Колмогорова: Dn={Dn:.4f}, Критич.={critical_value:.4f}')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True)
plt.savefig('kolmogorov_test.png')
plt.show()

print(f"Статистика Dn = {Dn:.6f}")
print(f"Критическое значение (alpha={alpha}) = {critical_value:.6f}")
print(f"Гипотеза H0 {'отвергается' if reject_h0 else 'не отвергается'} на уровне значимости alpha={alpha}")
print(f"p-value = {p_value:.6f}")