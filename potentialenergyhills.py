import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

k = 1  # spring constant
l0 = 1  # rest length!!
force = 0  # applied in newtons
mass = 0.1

# runtime variables
x = 0.7

y_start = -1
y_end = 1

discretizations = 500

zeros = np.empty((discretizations, 3))

dist = np.linspace(y_start, y_end, discretizations)

graph = force * dist
for ind, y in enumerate(dist):
    graph[ind] += k * (np.sqrt(x**2 + y**2) - l0) ** 2


plt.plot(dist, graph, color="black")

mins, _ = find_peaks(-graph)
plt.scatter(dist[mins], graph[mins], color="black")
maxes, _ = find_peaks(graph)
plt.scatter(dist[maxes], graph[maxes], color="red")

plt.show()
