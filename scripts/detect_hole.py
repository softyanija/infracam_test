import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import cv2

grayimg0 = cv2.imread('images/image0_19.png', cv2.IMREAD_GRAYSCALE)
#img1 = cv2.imread('test_b.png', cv2.IMREAD_GRAYSCALE)


equ0 = cv2.equalizeHist(grayimg0)
ret, thresh0 = cv2.threshold(equ0, 50, 255, cv2.THRESH_BINARY)

colorimg0 = cv2.cvtColor(thresh0, cv2.COLOR_GRAY2BGR)

contours, hierarchy = cv2.findContours(thresh0, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])

        colorimg0 = cv2.ellipse(colorimg0, ellipse, (255, 0, 0), 2)
        cv2.drawMarker(colorimg0, (cx,cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

cv2.imshow("original_0", grayimg0)
cv2.imshow("equqlize_0", equ0)
cv2.imshow("thresh_0", thresh0)
cv2.imshow("hole_0", colorimg0)
cv2.waitKey(0)
cv2.destroyAllWindows()
