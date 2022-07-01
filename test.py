import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image and invert so blob is white on black background
img = cv2.imread('maps/mymap_inv.png',0)

# do some eroding of img, but not too much
kernel = np.ones((3,3), np.uint8)
img = cv2.erode(img, kernel, iterations=5)

# # threshold img
ret, thresh = cv2.threshold(img,127,255,0)

# do distance transform
dist = cv2.distanceTransform(thresh, distanceType=cv2.DIST_L2, maskSize=5)

# set up cross for tophat skeletonization
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
skeleton = cv2.morphologyEx(dist, cv2.MORPH_TOPHAT, kernel)

# threshold skeleton
ret, skeleton = cv2.threshold(skeleton,0,255,0)

# display skeleton
# ax = plt.subplot(1,1,1)
# ax.imshow(skeleton,cmap='Greys')
# # ax.legend()
# plt.title('Map + Contours')
# plt.xlabel('x, in px')
# plt.ylabel('y, in px')
# plt.show()

# # save results
cv2.imwrite('tall_blob_skeleton.png', skeleton)