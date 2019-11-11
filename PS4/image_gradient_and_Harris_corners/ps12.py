import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

min_d = sys.maxsize
max_d  = -sys.maxsize -1



#function thresholds the image to values between 0 and 255
#input  :     image , maximum value in the image
#output :     thresholded image


def threshold_image(img,max_value):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] *=(255/max_value)
    return img


def wait():
    cv.waitKey(0)

#function to display the image
#input  :     string for the window , image to display
#output :     none

def display_image( description, img):
    cv.imshow(description, img)
    cv.waitKey(0)

def save_image(description,img):
    file_name =description + ".png"
    cv.imwrite(file_name, img)


#function to display 3d plot of the hough transform
#input  :     hough accumulator array
#output :     none


def plot_3d_graphs(hough_space, figname):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, hough_space.shape[0], 1)
    Y = np.arange(0,hough_space.shape[1],1)
    X,Y = np.meshgrid(X,Y)
    Z = hough_space[X,Y]
    zmax = np.amax(hough_space)
    zmin = np.amin(hough_space)
    plt.xlabel('d values')
    plt.ylabel('theta values')
    plt.suptitle('hough space')


    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z,rcount=hough_space.shape[0],ccount=hough_space.shape[1],cmap='PuBu',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.contour3D(X, Y, Z,rcount=591,ccount=181,cmap='Greens')
    ax.set_zlim(zmin, zmax)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    fig.savefig(figname)



#function to detect lines in the hough space
#input  :     image, threshold (above which the elements will be considered)
#output :     image with lines drawn on it


def detect_line_in_image(hough_space,img,threshold):

    image_size_x = img.shape[0]
    image_size_y=img.shape[1]
    maximum_distance = int(np.sqrt(image_size_x*image_size_x + image_size_y*image_size_y))
    offset=maximum_distance
    #extracting the maximum values form the hough space by seeing the
    limit = (int)(threshold*255)
    print(limit)
    for d in range(hough_space.shape[0]):
        for angle in range(hough_space.shape[1]):
            if(hough_space[d,angle] >limit):
                if (angle ==0):
                    #print("got zero")
                    y_value = d-offset
                    cv.line(img,(0,y_value), (img.shape[1],y_value),(0,255,0),2)
                    continue
                elif(angle == 90):
                    #print("got 90")
                    x_value=d-offset
                    cv.line(img, (x_value,0), (x_value,img.shape[0]),(0,255,0),2)
                    continue
                else:

                    theta = np.deg2rad(angle)
                    r = d-offset
                    for x_value in range(0,img.shape[1],2):
                        x1 =x_value
                        x2 = x_value+1
                        y1 = int((r - x1*np.cos(theta))/np.sin(theta))
                        y2 = int((r - x2*np.cos(theta))/np.sin(theta))
                        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    return img



#function to calculate the hough transform
#input  :     edge_points,  original image
#output :     hough accumulator array




def hough_transform (dst,img):

    #variable declared and defined here
    min_d = sys.maxsize
    max_d = -sys.maxsize-1

    #finding the edge points in the gradient whose value is greater than 0 and calculating thier index
    x,y = dst.nonzero()
    #size_x = dst.shape[0]
    image_size_x = img.shape[1]
    image_size_y = img.shape[0]

    theta_values = np.deg2rad(np.arange(0,180,1))
    no_of_theta = len(theta_values)

    sin_theta_values = np.sin(theta_values)
    cos_theta_values = np.cos(theta_values)
    maximum_distance = int(np.sqrt(image_size_x*image_size_x + image_size_y*image_size_y))
    offset=maximum_distance
    maximum_bin_size=0

    #declaring the hough space based on the minimum and the maximum values
    hough_space = np.zeros((2*maximum_distance,no_of_theta), dtype=np.float32)
    for i in range(len(x)):
        for angle in range(no_of_theta):
            d = int(y[i]*cos_theta_values[angle] + x[i]*sin_theta_values[angle])
            d= d+offset
            hough_space[d][angle]+=1
            if(hough_space[d][angle] > maximum_bin_size):
                maximum_bin_size= hough_space[d][angle]






    #thresholding the hough hough_space
    hough_space = threshold_image(hough_space,maximum_bin_size)
    return hough_space


    # hough_space array contains the values of the hough transform
    #plot_graphs_3d(hough_space,min_d,max_d)
