import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, BboxPatch, BboxConnector


def mark_inset(parent_axes, inset_axes, loc1, loc2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
    fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
    pp = BboxPatch(rect, fill=fill, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1, loc2=4 - loc1 + 1, **kwargs)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2, loc2=4 - loc2 + 1, **kwargs)
    inset_axes.add_patch(p1)
    inset_axes.add_patch(p2)
    p1.set_clip_on(False)
    p2.set_clip_on(False)

    return pp, p1, p2


img1 = cv2.imread('result3.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('result4.png', cv2.IMREAD_GRAYSCALE)

(x1, y1), (x2, y2) = (300, 220), (420, 330)  # right eye
# (x1, y1), (x2, y2) = (85, 260), (220, 370)  # left eye

img1_zoomed = np.zeros(img1.shape)
img2_zoomed = np.zeros(img2.shape)
img1_zoomed[y1:y2, x1:x2] = img1[y1:y2, x1:x2]
img2_zoomed[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img1, cmap='gray')
axes[1].imshow(img2, cmap='gray')
axes[0].set_title('Floyd-Steinberg')
axes[1].set_title('Jarvis')

axz0 = zoomed_inset_axes(axes[0], zoom=2, loc=4)
axz1 = zoomed_inset_axes(axes[1], zoom=2, loc=4)
axz0.imshow(img1_zoomed, cmap='gray')
axz1.imshow(img2_zoomed, cmap='gray')
axz0.tick_params(labelleft=False, labelbottom=False)
axz1.tick_params(labelleft=False, labelbottom=False)

axz0.set_xlim(x1, x2)
axz0.set_ylim(y2, y1)
axz1.set_xlim(x1, x2)
axz1.set_ylim(y2, y1)
mark_inset(axes[0], axz0, loc1=1, loc2=3, fc="none", ec="0.5")
mark_inset(axes[1], axz1, loc1=1, loc2=3, fc="none", ec="0.5")

plt.tight_layout()
# plt.show()
plt.savefig('report_images/cat-compare.png', dpi=600, bbox_inches='tight')
