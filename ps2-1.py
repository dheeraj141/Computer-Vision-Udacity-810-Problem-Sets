import numpy as np
import cv2 as cv
import ps12 as ps
import sys
import math
import multiprocessing as mp


#main function to calculate SSD
def check_img(img):
    if img is None:
        return 0
    else:
        return 1


def calculate_error(x1,x2):
    distance = 0
    for j in range(len(x1)):
        temp = int(int(x1[j]) - int(x2[j]))
        distance +=math.pow(temp,2)
    return distance

def check_var(x, left_image):
    if(x <0):
        x= 0
    elif (x >=left_image.shape[1]):
        x = left_image.shape[1] -1
    return x
def fill_values(left_gray,x, y,window_size):
    values = [0]*window_size
    size = int((window_size -1)/2)
    for j in range(1,size+1):


        values[check_var(j+size, left_gray)] = left_gray[x,check_var(y+j, left_gray)]
        values[check_var(size-j, left_gray)] = left_gray[x,check_var(y-j, left_gray)]
    values[size] = left_gray[x,y]
    return values







# function to calculate the SSD of the two images
# inputs : left_image , right_img  direction to calculate SSD
# Direction 0 => from left to right and 1 means right to left_image
# window size should be odd for symmetry
# output : return the disparity map


def calculate_SSD(left_image, right_image,direction, window_size,max_disparity):
    e1 = cv.getTickCount()
    left = np.asarray(left_image)
    right = np.asarray(right_image)

    window_size_half = int(window_size/2)
    disparity_left =np.zeros((left_image.shape[0],left_image.shape[1]))
    #breakpoint()
    for i in range(window_size_half,left_image.shape[0] - window_size_half):
        #l = [0]*left_image.shape[1]
        for j in range(window_size_half,left_image.shape[1] - window_size_half):


            min_distance = 65535
            min_j  = 0

            for disparity in range(max_disparity):
                distance = 0
                temp =0;
                for l in range(-window_size_half, window_size_half):
                    for m in range(-window_size_half, window_size_half):
                        if(direction == 0):
                             # meaning calculating disparity for left image
                             temp= int(left[i+l, j+m]) -  int(right[i+l, (j+m)-disparity])
                        else:
                            temp= int(right[i+l, j+m]) -  int(left[i+l, (j+m)+disparity])
                        distance += temp*temp
                if (distance <min_distance):
                    min_distance=distance
                    min_j = disparity
            #breakpoint()
            disparity_left[i,j] = min_j

    e2 = cv.getTickCount()
    print("time taken is ", (e2-e1)/cv.getTickFrequency())
    ps.display_image("disparity image", disparity_left)
    #ps.display_image("disparity_right", disparity_right)
    return disparity_left









def main(argv):
    if(len(argv) < 1):
        print("not enough parameters\n")
        print("usage PS1-1.py <path to image>\n")
        return -1


    left_image = cv.imread(argv[0],cv.IMREAD_COLOR)
    right_image = cv.imread(argv[1], cv.IMREAD_COLOR)

    x = check_img(left_image)
    y = check_img(right_image)
    if (x == 0 or y == 0):
        print("Error opening image\n")
        return -1

    ps.display_image("Image", left_image)
    left_gray = cv.cvtColor(left_image,cv.COLOR_BGR2GRAY)
    ps.display_image("gray image ",left_gray)
    right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    ps.display_image("graY image right", right_gray)
    #img = calculate_SSD(left_gray,right_gray,0,5)
    #disparity_left = left_gray.copy()



    disparity_left = calculate_SSD(left_gray,right_gray,1,6,30)
    disparity_left = ps.threshold_image(disparity_left,30)
    ps.display_image("disparity image", disparity_left)
    ps.save_image("disparity_image", disparity_left)







# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
