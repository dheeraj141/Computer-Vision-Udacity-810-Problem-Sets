import numpy as np
import cv2 as cv
import ps12 as ps
import sys
import math
from operator import itemgetter
def min(x,index):
    # check that the index is within the size of the array
    min = sys.maxsize
    for j in x:
        if(min >j[index]):
            min =j[index]
    return min
def max(x,index):
    # check that the index is within the size of the array
    max = -sys.maxsize -1
    for j in x:
        if(max <j[index]):
            max =j[index]
    return max
def threshold_image_circle(img,max_value):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i][j][k] *=(255/max_value)
    return img
def calculate_distance(x2,x1):
    x_distance = pow((x2[0]-x1[0]),2)
    y_distance = pow((x2[1]-x1[1]),2)

    distance  =math.sqrt(x_distance + y_distance)
    return distance

def  refine_circle_centers(x):
    x.sort(key=lambda tup: tup[2])
    minimum_distance  = int(input("Enter the minimum distance between circels\n"))
    j =0;
    result = []
    result.append(j)
    while(j <len(x)):

        k =j+1
        while(k <len(x) and calculate_distance(k,j) <minimum_distance):
            k+=1

        result.append(k)
        j = k+1
    return result





def detect_circles_in_hough_space(hough_space, threshold,offset_x,offset_y,offset_r):
    result = []
    prev = (0,0,0)
    limit = int(threshold*255)
    for i in range(hough_space.shape[0]):
        for j in range(hough_space.shape[1]):
            for k in range(hough_space.shape[2]):
                if(hough_space[i][j][k] >limit):
                    x =(i+ offset_x,j+offset_y,k+offset_r)
                    result.append(x)
    result1 =refine_circle_centers(result)
    return result1





def hough_space_circle(edge_array,rmin,rmax):
    min_x = min(edge_array,1)
    min_y = min(edge_array,0)
    max_x = max(edge_array,1)
    max_y = max(edge_array,0)
    size_x = max_x - min_x +  2*rmax+1
    size_y = max_y- min_y + 2*rmax
    size_z = rmax -rmin+1
    offset_x = min_x - rmax
    offset_y = min_y - rmax
    hough_accumulator = np.zeros((size_x,size_y,size_z))
    maximum_accumulator_value =0

    for j in edge_array:
        for r in range(rmin,rmax):

            for theta in range(0,361,2):
                a = int(j[1] - r*np.cos(theta *(math.pi/180)))
                b = int(j[0] - r*np.sin(theta*(math.pi/180)))
                hough_accumulator[a-offset_x,b-offset_y,r-rmin]+=1;
                if(maximum_accumulator_value<hough_accumulator[a-offset_x,b-offset_y,r-rmin]):
                    maximum_accumulator_value = hough_accumulator[a-offset_x,b-offset_y,r-rmin]
    hough_space = threshold_image_circle(hough_accumulator,maximum_accumulator_value)
    threshold = float(input("enter the threshold value\n"))
    result = detect_circles_in_hough_space(hough_space,threshold,offset_x,offset_y,rmin)
    return result




def extract_edges(edge_image):
    return [(x,y) for x in range(edge_image.shape[0]) for y in range(edge_image.shape[1]) if edge_image[x,y] ==255]



def main(argv):

    if(len(argv) < 1):
        print("not enough parameters\n")
        print("usage PS1-1.py <path to image>\n")
        return -1


    img = cv.imread(argv[0],cv.IMREAD_COLOR)
    if img is None:
        print("Error opening image\n")
        return -1





    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_smooth = cv.GaussianBlur(gray,(5,5),4,4)



    edges = cv.Canny(gray_smooth,80,180)
    ps.display_image("edges", edges)
    ps.save_image("ps1-5-a", edges)
    x = extract_edges(edges)
    rmin = int(input("enter the minimum raidus\n"))
    rmax = int(input("enter the max radius\n"))
    circles_in_image = hough_space_circle(x,rmin, rmax)
    #centers = detecting_circle_using_chord(x)
    for j in circles_in_image:
        if (j[2] >=rmin and j[2] <rmax):
            cv.circle(img,(j[0],j[1]), j[2], (0,255,0))
    ps.display_image("cirlces in image",img)
    ps.save_image("ps1-5-b", img)
    return 0
# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
