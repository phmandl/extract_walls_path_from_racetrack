#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

class find_path:
    def __init__(self,map_name,map_reso,step_size,plot_flag = True) -> None:
        self.file_path = os.path.join(os.getcwd(), 'maps', map_name)
        self.reso = map_reso
        self.step_size = step_size
        self.plot_flag = plot_flag

        self.max_steps = 500
        self.outer_cont = 0
        self.inner_cont = 1
        
        # Read the original image
        self.img = cv2.imread(self.file_path)
        self.img = cv2.flip(self.img,0) # flip horizontal for TUM Tool 

        # Convert to graycsale
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Blur the image for better edge detection
        self.img_blur = cv2.GaussianBlur(self.img_gray, (5,5), 0)

        # apply binary thresholding
        self.ret, self.thresh = cv2.threshold(self.img_blur, 127, 255, cv2.THRESH_BINARY) 

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        self.contours, self.hierarchy = cv2.findContours(image=self.thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # Save vector
        self.cont_in = []
        self.cont_out = []
        self.midPoint = []
        self.disToWall = []

    def interp_cont(self,idx):
        x = np.squeeze(self.contours[idx])[:,0]
        y = np.squeeze(self.contours[idx])[:,1]
        points = np.array([x,y]).T

        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0)/distance[-1]

        # Interpolation for different methods:
        interpolations_methods = ['slinear', 'quadratic', 'cubic']
        alpha = np.linspace(0, 1, 100)

        interpolator =  interp1d(distance, points, kind=interpolations_methods[2], axis=0)
        return interpolator(alpha)

    def plot_map_contours(self):
        ax = plt.subplot(1,1,1)
        ax.imshow(self.img,cmap='Greys')
        for idx in np.arange(0,len(self.contours)):
            cont = np.squeeze(self.contours[idx])
            xDat = cont[:,0]
            yDat = cont[:,1]
            ax.plot(xDat,yDat,label=str(idx))
        ax.legend()
        plt.title('Map + Contours')
        plt.xlabel('x, in px')
        plt.ylabel('y, in px')
        plt.show()

    def calc_path(self):
        self.cont_out = self.interp_cont(self.outer_cont)
        self.cont_out = np.vstack([self.cont_out,self.cont_out[0,:]]) # Close loop
        
        self.cont_in = self.interp_cont(self.inner_cont)
        self.cont_in = np.vstack([self.cont_in,self.cont_in[0,:]]) # Close loop



        ax = plt.subplot(1,1,1)
        ax.imshow(self.thresh,cmap='Greys')
        ax.plot(self.cont_out[:,0],self.cont_out[:,1],label='Outer')
        ax.plot(self.cont_in[:,0],self.cont_in[:,1],label='Inner')
        ax.legend()
        plt.title('Map + Contours')
        plt.xlabel('x, in px')
        plt.ylabel('y, in px')
        plt.show()


       
       
        out = cv2.distanceTransform(self.thresh, distanceType = cv2.DIST_L2, maskSize = 0)
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        out = cv2.filter2D(out, -1, kernel)

        idx_range = np.arange(-2,3,1)
        print(idx_range)

        for idx,val in enumerate(np.squeeze(self.contours[0])):
            y_val = val[0]
            x_val = val[1]

            out[x_val:x_val+6,y_val:y_val+6] = 0

        for idx,val in enumerate(np.squeeze(self.contours[1])):
            y_val = val[0]
            x_val = val[1]

            out[x_val:x_val+6,y_val:y_val+6] = 0

            # out[x_val,y_val] = 0

        # print(out.shape)
        # print(out)
 
        # out[out < 0.9995] = 0
        out = cv2.GaussianBlur(out, (9,9), 0)
        # out[out < 0.9995] = 0

        kernel = np.ones((2, 2), np.uint8)
  
        # Using cv2.erode() method 
        out = cv2.erode(out, kernel)

        # apply binary thresholding
        ret, out = cv2.threshold(out, 0.9, 1, cv2.THRESH_BINARY) 

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        # contours, hierarchy = cv2.findContours(image=out, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)








        ax = plt.subplot(1,1,1)
        ax.imshow(out,cmap='Greys')
        # ax.plot(self.cont_out[:,0],self.cont_out[:,1],label='Outer')
        # ax.plot(self.cont_in[:,0],self.cont_in[:,1],label='Inner')
        ax.legend()
        plt.title('Map + Contours')
        plt.xlabel('x, in px')
        plt.ylabel('y, in px')
        plt.show()

        


    def save_to_csv(self,fileName):
        a = np.asarray((self.midPoint[:,0],self.midPoint[:,1],self.disToWall[:,0],self.disToWall[:,0]))
        a = np.transpose(a)
        np.savetxt(fileName, a,  delimiter=',', comments="")


if __name__ == "__main__":
    map_to_optimize = find_path(map_name='mymap_inv.png', map_reso=0.05, step_size= 0.5)
    map_to_optimize.calc_path()
    