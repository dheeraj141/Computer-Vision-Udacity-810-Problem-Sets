import numpy as np
import cv2 as cv
import ps12 as ps
import sys
import math

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


def detect_circles_in_hough_space(hough_space, threshold,offset_x,offset_y,offset_r):
    result = []
    limit = int(threshold*255)
    for i in range(hough_space.shape[0]):
        for j in range(hough_space.shape[1]):
            for k in range(hough_space.shape[2]):
                if(hough_space[i][j][k] >limit):
                    x =(i+ offset_x,j+offset_y,k+offset_r)
                    result.append(x)
    return result





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
    result = detect_circles_in_hough_space(hough_space,0.5,offset_x,offset_y,rmin)
    return result
def detecting_circle_using_gradient(edge_points):




    max =0
    centers =[]
    limit = 150
    min_a = sys.maxsize
    max_a = -sys.maxsize-1
    min_b = sys.maxsize
    max_b = -sys.maxsize-1
    for j in range(len(edge_array)):
        for k in range(j,len(edge_array)):
            denominator = (edge_array[j][1] -edge_array[k][1])
            if(denominator>0):
                a = int((edge_array[j][0] -edge_array[k][0])/denominator)
                b = math.pow(edge_array[j][0],2) + math.pow(edge_array[j][1],2) - math.pow(edge_array[k][0],2) - math.pow(edge_array[k][1],2)
                b = int(b/2*denominator)
                if(min_a>a):
                    min_a = a
                if(max_a <a):
                    max_a = a
                if(min_b>b):
                    min_b = b
                if(max_b <b):
                    max_b = b
                temp =(a,b)
                result.append(temp)
    if(min_a <0):
        size_a = max_a+(-min_a)+1
    else:
        size_a = max_a -min_a +1
    if(min_b <0):
        size_b = max_b+(-min_b)+1
    else:
        size_b = max_b -min_b+1
    hough_circle_chord = np.zeros((size_a,size_b))
    print(size_a,size_b)
    print(len(result))

    for j in result:
        hough_circle_chord[j[0]-min_a][j[1]-min_b]+=1






    for i in range(size_a):
        for j in range(size_b):
            if(hough_circle_chord[i,j] >limit):
                temp =(i+min_a,j+min_b)
                centers.append(temp)
    return centers


def extract_edges(edge_image):
    return [(x,y) for x in range(edge_image.shape[0]) for y in range(edge_image.shape[1]) if edge_image[x,y] ==255]

img = cv.imread('../image/ps1-input1.jpg', cv.IMREAD_COLOR)
monochrome_image = img[:,:,1]
ps.display_image("original image", img)
ps.display_image("monochrome image", monochrome_image)

monochrome_smooth_image = cv.GaussianBlur(monochrome_image,(5,5),4,4)
ps.display_image("smoothened  monochrome image", monochrome_smooth_image)


edges_smooth_monochrome = cv.Canny(monochrome_smooth_image,100,210)
ps.display_image("edges in the smoothed monochrome image", edges_smooth_monochrome)
x = extract_edges(edges_smooth_monochrome)
circles_in_image = hough_space_circle(x,1,10)
#centers = detecting_circle_using_chord(x)
for j in circles_in_image:
    cv.circle(img,(j[0],j[1]), j[2], (0,255,0))
ps.display_image("cirlces in image",img)
