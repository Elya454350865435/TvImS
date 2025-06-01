import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46,
    9.63, 4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26,
    -17.19, 9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14,
    -8.98, -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68
])

h = 4.0

variational_series = np.sort(data)
print("Вариационный ряд:")
print(variational_series)

def ecdf(x):
    return np.searchsorted(variational_series, x, side='left') / len(data)

x_plot = np.linspace(np.min(data) - 1, np.max(data) + 1, 500)
y_plot = np.array([ecdf(x) for x in x_plot])

plt.figure(figsize=(10, 5))
plt.step(x_plot, y_plot, where='post', label='Эмпирическая функция распределения')
plt.title('Эмпирическая функция распределения')
plt.xlabel('Значения выборки')
plt.ylabel('F(x)')
plt.grid(True)
plt.legend()
plt.savefig('ecdf_plot.png')
plt.show()

bins = np.arange(np.floor(data.min()), np.ceil(data.max()) + h, h)
counts, bin_edges = np.histogram(data, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], counts, width=h, align='edge', alpha=0.7,
        edgecolor='black', label='Гистограмма')
plt.plot(bin_centers, counts, 'ro-', label='Полигон частот')
plt.title(f'Гистограмма и полигон частот (шаг h = {h})')
plt.xlabel('Интервалы')
plt.ylabel('Частота')
plt.xticks(bin_edges, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('histogram_polygon.png')
plt.show()

print("\nТаблица частот:")
for i in range(len(bin_edges) - 1):
    print(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {counts[i]}")

print("\nЭмпирическая функция распределения:")
n = len(data)
print(f"Для x <= {variational_series[0]:.3f}, F_n(x) = {0.0:.3f}")

for i in range(len(variational_series)-1):
    current = variational_series[i]
    next_val = variational_series[i+1]
    if current == next_val:
        continue
    f_value = (i+1) / n
    print(f"Для {current:.3f} < x <= {next_val:.3f}, F_n(x) = {f_value:.3f}")

print(f"Для x > {variational_series[-1]:.3f}, F_n(x) = {1.0:.3f}")