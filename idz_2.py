import numpy as np

data = np.array([
    4.02, 7.36, 5.85, -6.78, -3.40, 0.83, -8.32, -8.45, -19.77, -12.57, -7.69, 8.46,
    9.63, 4.85, -3.21, 3.88, -24.86, -3.46, -16.44, -3.26, -19.02, -22.53, -10.05, -4.26,
    -17.19, 9.63, 3.52, -0.05, -9.78, -9.57, -3.40, -3.17, 0.38, -2.33, -15.15, 21.99, 6.14,
    -8.98, -13.79, -4.64, 6.75, 5.99, 10.89, 3.93, 7.22, 10.99, 1.25, -14.13, -7.03, -2.68
])

c = -9.00
d = 5.00
n = len(data)

mean = np.sum(data) / n
squared_deviations = np.sum((data - mean) ** 2)
variance = squared_deviations / n
std_dev = np.sqrt(variance)

sorted_data = np.sort(data)
median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2

skewness = np.sum((data - mean) ** 3) / n / (std_dev ** 3)
kurtosis = np.sum((data - mean) ** 4) / n / (std_dev ** 4) - 3

in_interval = np.sum((data >= c) & (data <= d))
prob = in_interval / n

print(f"(a) Математическое ожидание: {mean:.6f}")
print(f"(b) Дисперсия: {variance:.6f}")
print(f"(c) СКО: {std_dev:.6f}")
print(f"(d) Медиана: {median:.6f}")
print(f"(e) Асимметрия: {skewness:.6f}")
print(f"(f) Эксцесс: {kurtosis:.6f}")
print(f"(g) Вероятность P(X ∈ [{c},{d}]): {prob:.6f}")