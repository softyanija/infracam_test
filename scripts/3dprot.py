import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import cv2

def get_histogram(img):
    channels = 1
    histogram = []
    for ch in range(channels):
        hist_ch = cv2.calcHist([img],[ch],None,[256],[0,256])
        histogram.append(hist_ch[:,0])

    return histogram

def draw_histogram(hist):
    ch = len(hist)
    if (ch == 1):
        colors = ["black"]
        label = ["Gray"]
    else:
        colors = ["blue", "green", "red"]
        label = ["B", "G", "R"]

    x = range(256)
    for col in range(ch):
        y = hist[col]
        plt.plot(x, y, color = colors[col], label = label[col])
    plt.legend(loc=2)
    plt.show()

img1 = cv2.imread('images/image0_19_crop.png', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('test_b.png', cv2.IMREAD_GRAYSCALE)
#reverse = cv2.bitwise_not(img1)
height, width = img1.shape[:2]
ch = 1
colors = ["black"]
label = ["Gray"]

hist_default = get_histogram(img1)
equ = cv2.equalizeHist(img1)
hist_equ = get_histogram(equ)


# grid data
Xa = np.linspace(0, width-1, width)
Ya = np.linspace(0, height-1, height)
xp, yp = np.meshgrid(Xa,Ya)

# grey-scale data
z = equ[yp.astype(np.int32), xp.astype(np.int32)]

# show graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=85, azim=90)
ax.plot_surface(xp, yp, z, rstride=1, cstride=1, cmap=cm.coolwarm)
ax.invert_xaxis()
ax.invert_zaxis()
ax.set_title('grayscale_3dprot')
plt.savefig("grayscale_3dprot_equ.png")
plt.show()

