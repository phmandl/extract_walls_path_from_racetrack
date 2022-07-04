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
        self.outer_cont = 2
        self.inner_cont = 3
        
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
        alpha = np.linspace(0, 1, 5)

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
        
    def my_interpolation(self,vec,increase_num):
        x_vec = np.linspace(0,1,len(vec))
        x_vec_new = np.linspace(0,1,len(vec)*increase_num)
        f =interp1d(x_vec, vec,kind='cubic')
        
        new_vec = f(x_vec_new)
        
        return new_vec

    def calc_path(self):
        self.cont_out = self.interp_cont(self.outer_cont)*self.reso
        self.cont_in = self.interp_cont(self.inner_cont)*self.reso
        
        
        faktor = 20
        # inner
        inner_x = self.cont_in[:,0]
        inner_y = self.cont_in[:,1]
        inner_x_new = self.my_interpolation(inner_x,faktor)
        inner_y_new = self.my_interpolation(inner_y,faktor)
        # outer
        outer_x = self.cont_out[:,0]
        outer_y = self.cont_out[:,1]
        outer_x_new = self.my_interpolation(outer_x,faktor)
        outer_y_new = self.my_interpolation(outer_y,faktor)
        
        #
        ax2 = plt.subplot(1,1,1)
        ax2.plot(inner_x_new,inner_y_new,marker='x')
        ax2.plot(outer_x_new,outer_y_new,marker='x')
        plt.show()
        
        self.cont_in = np.zeros((len(outer_y_new),2))
        self.cont_in[:,0] = inner_x_new
        self.cont_in[:,1] = inner_y_new
        
        self.cont_out = np.zeros((len(outer_y_new),2))
        self.cont_out[:,0] = outer_x_new
        self.cont_out[:,1] = outer_y_new
        
        # print(np.vstack((inner_x,inner_y)))
        # print(self.cont_in)
        # self.cont_in = np.vstack((inner_x,inner_y))
        # self.cont_out = np.vstack((outer_x,outer_y))
        
        
        # self.cont_out = np.squeeze(self.contours[self.outer_cont])*self.reso
        # self.cont_in = np.squeeze(self.contours[self.inner_cont])*self.reso

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
            ax.scatter(middle_point[0],middle_point[1],marker='x',color='black')
            ax.plot(normal_dir[:,0],normal_dir[:,1],color='orange') 

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
                ax.scatter(middle_point[0],middle_point[1],marker='x',color='black')
                ax.plot(normal_dir[:,0],normal_dir[:,1],color='orange')
        
        if self.plot_flag == True:
            ax.legend()
            ax.axis('equal')
            plt.show()
        
        # ----------------------------------------------------------
        # interpolation 
        middle_line = self.midPoint[:-1,:]
        middle_line = np.vstack((middle_line[-2:-1,:],middle_line,middle_line[0:2,:]))
        
        num_to_increse=5
        #print(middle_line)
        middle_x = self.my_interpolation(middle_line[:,0], num_to_increse)
        middle_y = self.my_interpolation(middle_line[:,1], num_to_increse)
        middle_x = middle_x[num_to_increse+1:-num_to_increse-1]
        middle_y = middle_y[num_to_increse+1:-num_to_increse-1]
        
        
        ax3 = plt.subplot()
        ax3.plot(inner_x_new,inner_y_new)
        ax3.plot(outer_x_new,outer_y_new)
        ax3.scatter(middle_line[:,0],middle_line[:,1])
        ax3.plot(middle_x,middle_y)
        
        
        
        
        [new_inner, new_outer, dist_to_inner, dist_to_outer] = self.normal_distance(middle_x,middle_y,inner_x_new,inner_y_new,outer_x_new,outer_y_new)
        self.dist_to_inner = dist_to_inner
        self.dist_to_outer = dist_to_outer
        self.middle_x = middle_x
        self.middle_y = middle_y
        print(dist_to_inner)
        
        
        ax3.plot(new_inner[0,:],new_inner[1,:])
        ax3.plot(new_outer[0,:],new_outer[1,:])
        ax3.set_aspect('equal', 'box')
        ax3.grid()
        plt.show()
        
        
        #print(new_inner)
        fig, ax4 = plt.subplots(figsize=(6,4),dpi=400)
        ax4.plot(middle_x,middle_y)
        for i in range(len(middle_x)):
            ax4.plot([new_outer[i,0],new_inner[i,0]],[new_outer[i,1],new_inner[i,1]],color='black',linewidth=0.5)
        ax4.set_aspect('equal', 'box')
        plt.show()
            
        
    def normal_distance(self,middle_x,middle_y,inner_x,inner_y,outer_x,outer_y):
        to_go = np.linspace(0.05,3,5000)
        list_temp = []
        N = len(middle_x)
        
        new_inner = np.zeros((len(middle_x),2))
        dist_to_inner = np.zeros((len(middle_x),1))
        new_outer = np.zeros((len(middle_x),2))
        dist_to_outer = np.zeros((len(middle_x),1))
        
        for i in range(N):
            # midpoint rule
            ind_min = (i-1)%N
            ind_max = (i+1)%N
            dx = middle_x[ind_max]-middle_x[ind_min]
            dy = middle_y[ind_max]-middle_y[ind_min]
            
            nx = +dy/np.sqrt(dx**2+dy**2)
            ny = -dx/np.sqrt(dx**2+dy**2)
            
            old = 5
            for n in to_go:
                dx = -inner_x + middle_x[i]+nx*n
                dy = -inner_y + middle_y[i]+ny*n
                ind_min_norm = np.linalg.norm(np.array([dx,dy]), axis=0).argmin()
                mindist= np.sqrt(dx[ind_min_norm]**2+dy[ind_min_norm]**2)
                if mindist>old:
                    break
                old = mindist
            new_inner[i,:]=[middle_x[i]+nx*n,middle_y[i]+ny*n]
            # print(nx,ny)
            dist_to_inner[i] = n
            
            old = 5
            for n in to_go:
                dx = -outer_x + middle_x[i]-nx*n
                dy = -outer_y + middle_y[i]-ny*n
                ind_min_norm = np.linalg.norm(np.array([dx,dy]), axis=0).argmin()
                mindist= np.sqrt(dx[ind_min_norm]**2+dy[ind_min_norm]**2)
                if mindist>old:
                    break
                old = mindist
            new_outer[i,:]=[middle_x[i]-nx*n,middle_y[i]-ny*n]
            dist_to_outer[i] = n
            

        return new_inner, new_outer, dist_to_inner, dist_to_outer

    def calc_line_out_in(self,idx_in,idx_out):
        return np.vstack((self.cont_out[idx_out],self.cont_in[idx_in]))

    def save_to_csv(self,fileName):
        #a = np.asarray((self.midPoint[:,0],self.midPoint[:,1],self.disToWall[:,0],self.disToWall[:,0]))
        
        a = np.asarray((self.middle_x[:],self.middle_y[:],self.dist_to_inner[:,0],self.dist_to_outer[:,0]))
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
    map_to_optimize = find_path(map_name='mymap.png', map_reso=0.05, step_size= 1.6)
    # map_to_optimize.plot_map_contours()
    map_to_optimize.calc_path()
    
    line = map_to_optimize.midPoint

    x = line[:,0]
    y = line[:,1]
    points = np.array([x,y]).T

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)/distance[-1]

    # Interpolation for different methods:
    interpolations_methods = ['slinear', 'quadratic', 'cubic']
    alpha = np.linspace(0, 1, 500)

    interpolator = interp1d(distance, points, kind='cubic', axis=0)

    line_intp = interpolator(alpha)

    ax = plt.subplot(1,1,1)
    ax.imshow(map_to_optimize.img,cmap='Greys')
    ax.plot(x,y)
    ax.plot(line_intp[:,0], line_intp[:,1])
    ax.legend()
    plt.title('Map + Contours')
    plt.xlabel('x, in px')
    plt.ylabel('y, in px')
    plt.show()


    map_to_optimize.save_to_csv(os.path.join('output','foo.csv'))

