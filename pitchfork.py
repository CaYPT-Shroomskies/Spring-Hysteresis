import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

k = 1 # spring constant
l0 = 1 # rest length!! 
force = -0.01 # applied in newtons
mass = 0.1

# runtime variables
x_start = 0.7
x_end = 1.2

y_start  = -1
y_end = 1

discretizations = 1000

zeros = np.empty((discretizations,3))
zeros[:] = np.nan

x_space = np.linspace(x_start,x_end,discretizations)
dist = np.linspace(y_start,y_end,discretizations)

flip = False
if force < 0:
    force *= -1
    flip = True
    
for (num,x) in enumerate(x_space):
    graph = force*dist
    for (ind,y) in enumerate(dist):
        graph[ind] += k*(np.sqrt(x**2+y**2)-l0)**2

    mins,_ = find_peaks(-graph)
    maxes,_ = find_peaks(graph)

    for (i,value) in enumerate(mins):
        zeros[num,i] = dist[value]
        
    for value in maxes:
        zeros[num,2] = dist[value]

if flip:
    zeros[:] *= -1
plt.plot(x_space,zeros[:,0],color="black")
plt.plot(x_space,zeros[:,1],color="black")
plt.plot(x_space,zeros[:,2],"--",color="black")


plt.show()
