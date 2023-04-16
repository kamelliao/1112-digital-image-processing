import matplotlib.pyplot as plt
import numpy as np

x = np.load('feat.npy')
y = np.random.rand(600*900)
lbl = np.load('labels.npy')
colors = np.load('color.npy')
data = [x[lbl==k] for k in range(lbl.max() + 1)]

fig, ax = plt.subplots()
bplot = ax.boxplot(data, vert=False, patch_artist=True)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel('X[0]')
ax.set_ylabel('cluster id')

ratio = .3
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.show()