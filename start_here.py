#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class find_path:
    def __init__(self,map_name,map_reso,step_size,plot_flag = True) -> None:
        self.file_path = os.path.join(os.getcwd(), 'maps', map_name)
        self.reso = map_reso
        self.step_size = step_size
        self.plot_flag = plot_flag

        self.max_steps = 1000
        self.outer_cont = 2
        self.inner_cont = 3
        
        # Read the original image
        self.img = cv2.imread(self.file_path) 

        # Convert to graycsale
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Blur the image for better edge detection
        self.img_blur = cv2.GaussianBlur(self.img_gray, (3,3), 0)

        # apply binary thresholding
        self.ret, self.thresh = cv2.threshold(self.img_blur, 127, 255, cv2.THRESH_BINARY) 

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        self.contours, self.hierarchy = cv2.findContours(image=self.thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # Save vector
        self.cont_in = []
        self.cont_out = []
        self.midPoint = []
        self.disToWall = []

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
        self.cont_out = np.squeeze(self.contours[self.outer_cont])*self.reso
        self.cont_in = np.squeeze(self.contours[self.inner_cont])*self.reso

        # Start point - start at first point of outher contour
        idx_out = 0
        idx_in = np.linalg.norm(self.cont_out[idx_out] - self.cont_in, axis=1).argmin()
        trans_line = self.calc_line_out_in(idx_in,idx_out)
        middle_point, dis_to_wall, new_point, normal_dir = self.calc_midPoint_normal(trans_line)

        # Save Stuff
        self.midPoint = middle_point
        self.disToWall = dis_to_wall

        if self.plot_flag == True:
            self.plot_map_contours() # Show map with all contours

            # New plot
            ax = plt.subplot(1,1,1)
            ax.plot(self.cont_out[:,0],self.cont_out[:,1 ],label='outer',color='green')
            ax.plot(self.cont_in[:,0],self.cont_in[:,1],label='inner',color='blue')

            # Add to plot each step
            ax.plot(trans_line[:,0],trans_line[:,1],color='red')
            ax.scatter(middle_point[0],middle_point[1])
            ax.plot(normal_dir[:,0],normal_dir[:,1],color='black') 

        idx_out_old = idx_out
        for idx in np.arange(0,self.max_steps):
            idx_out = np.linalg.norm(new_point - self.cont_out,axis=1).argmin() 
            idx_in = np.linalg.norm(new_point - self.cont_in,axis=1).argmin() 

            # Check stopping condition
            if idx_out_old > idx_out:    
                print("Break Index: ", idx)
                break
            idx_out_old = idx_out

            trans_line = self.calc_line_out_in(idx_in,idx_out)
            middle_point, dis_to_wall, new_point, normal_dir = self.calc_midPoint_normal(trans_line)

            # Save Stuff
            self.midPoint = np.vstack((self.midPoint,middle_point))
            self.disToWall = np.vstack((self.disToWall,dis_to_wall))

            
            if self.plot_flag == True:
                # Add to plot each step
                ax.plot(trans_line[:,0],trans_line[:,1],color='red')
                ax.scatter(middle_point[0],middle_point[1])
                ax.plot(normal_dir[:,0],normal_dir[:,1],color='black') 

        if self.plot_flag == True:
            ax.legend()
            ax.axis('equal')
            plt.show()

    def calc_line_out_in(self,idx_in,idx_out):
        return np.vstack((self.cont_out[idx_out],self.cont_in[idx_in]))

    def save_to_csv(self,fileName):
        a = np.asarray((self.midPoint[:,0],self.midPoint[:,1],self.disToWall[:,0],self.disToWall[:,0]))
        a = np.transpose(a)
        np.savetxt(fileName, a,  delimiter=',', comments="")

    def calc_midPoint_normal(self,line):
            dx = line[1,0] - line[0,0]
            dy = line[1,1] - line[0,1]

            middle_point = np.zeros(2)
            middle_point[0] = line[0,0] + dx/2
            middle_point[1] = line[0,1] + dy/2
            
            dis_to_wall = np.linalg.norm([dx/2,dy/2])

            new_point = np.zeros(2)
            nx = -dy/np.linalg.norm([dx,dy])
            ny = dx/np.linalg.norm([dx,dy])

            new_point = middle_point + np.array([nx,ny])*self.step_size

            normal_dir = np.array([middle_point,new_point])

            return middle_point, dis_to_wall, new_point, normal_dir

if __name__ == "__main__":
    map_to_optimize = find_path(map_name='f1_esp.png', map_reso=0.05, step_size= 1.0)
    map_to_optimize.calc_path()
    map_to_optimize.save_to_csv(os.path.join('output','foo.csv'))

