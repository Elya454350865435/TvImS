import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46,
    9.63, 4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26,
    -17.19, 9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14,
    -8.98, -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68
])
n = len(data)
a0 = -36.00
sigma0 = 10.00
alpha = 0.20

def phi(x):
    return norm.cdf(x)


def F0(x):
    return phi((x - a0) / sigma0)

data_sorted = np.sort(data)

print("=" * 85)
print(f"{'Критерий Колмогорова-Смирнова для проверки нормальности':^85}")
print(f"{'Параметры: a0 = ' + str(a0) + f', σ0 = {sigma0}, α = {alpha}, n = {n}':^85}")
print("=" * 85)
print(
    f"{'i':>2} | {'(i-1)/n':>8} | {'i/n':>8} | {'X(i)':>9} | {'F0(Xi)':>9} | {'Нижнее':>8} | {'Верхнее':>8} | {'Макс.откл.':>10}")
print(f"{'':>2} | {'':>8} | {'':>8} | {'':>9} | {'':>9} | {'отклон.':>8} | {'отклон.':>8} | {'в точке':>10}")
print("-" * 85)

Dn = 0.0
max_deviation_index = 0
results = []

for i, x in enumerate(data_sorted, 1):
    F_emp_prev = (i - 1) / n
    F_emp_curr = i / n
    F0_x = F0(x)

    lower_dev = F0_x - F_emp_prev  # F0 - Fn-
    upper_dev = F_emp_curr - F0_x  # Fn+ - F0
    current_max = max(abs(lower_dev), abs(upper_dev))

    if current_max > Dn:
        Dn = current_max
        max_deviation_index = i

    results.append((i, x, F_emp_prev, F_emp_curr, F0_x, lower_dev, upper_dev, current_max))

    print(f"{i:2d} | {F_emp_prev:8.5f} | {F_emp_curr:8.5f} | {x:9.5f} | {F0_x:9.5f} | "
          f"{lower_dev:8.5f} | {upper_dev:8.5f} | {current_max:10.5f}")

print("-" * 85)

critical_value = 1.07 / np.sqrt(n)
Dn_stat = np.sqrt(n) * Dn

print(f"\nМаксимальное отклонение D_n = {Dn:.6f}")
print(f"Статистика критерия √n * D_n = {Dn_stat:.6f}")
print(f"Критическое значение при α={alpha}: {critical_value:.6f}")

if Dn > critical_value:
    print("=> Гипотеза H₀ отвергается на уровне значимости α=0.20")
else:
    print("=> Нет оснований отвергнуть гипотезу H₀ на уровне значимости α=0.20")


def kolmogorov_pvalue(tau, terms=100):
    """Вычисление p-value для статистики Колмогорова"""
    p_val = 0.0
    for k in range(1, terms + 1):
        term = (-1) ** (k - 1) * math.exp(-2 * k ** 2 * tau ** 2)
        p_val += term
    return 2 * p_val


p_value = kolmogorov_pvalue(Dn_stat)
print(f"p-value ≈ {p_value:.10f}")

plt.figure(figsize=(12, 8))
x_range = np.linspace(-30, 25, 1000)
F0_vals = [F0(x) for x in x_range]

ecdf = np.arange(1, n + 1) / n
plt.step(np.concatenate([[-30], data_sorted, [25]]),
         np.concatenate([[0], ecdf, [1]]),
         where='post', label='Эмпирическая Fn(x)', linewidth=2)

plt.plot(x_range, F0_vals, 'r-', label='Теоретическая F0(x)', linewidth=2)

plt.fill_between(x_range, F0_vals - critical_value, F0_vals + critical_value,
                 color='gray', alpha=0.3, label='Доверительная область')

max_dev_x = data_sorted[max_deviation_index - 1]
plt.axvline(x=max_dev_x, color='g', linestyle='--',
            label=f'Макс. отклонение (x={max_dev_x:.2f})')

plt.title(f'Критерий Колмогорова: Dn={Dn:.4f}, Крит.={critical_value:.4f}', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('F(x)', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('kolmogorov_test.png', dpi=300)
plt.show()
