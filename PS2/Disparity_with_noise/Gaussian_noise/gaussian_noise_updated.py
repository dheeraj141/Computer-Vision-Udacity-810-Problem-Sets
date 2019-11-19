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


def add_gaussian_noise(mean, sd, left_image):
    h,w = left_image.shape
    noise = np.random.normal(mean, sd, (h,w))
    left = np.asarray(left_image)
    left = left.astype('float64')
    left+=noise
    left =left.astype('uint8')
    return left









# function to calculate the SSD of the two images
# inputs : left_image , right_img  direction to calculate SSD
# Direction 0 => from left to right and 1 means right to left_image
# window size should be odd for symmetry
# output : return the disparity map


def calculate_SSD(left_image, right_image,direction, window_size,max_disparity):
    e1 = cv.getTickCount()
    left = np.asarray(left_image)
    right = np.asarray(right_image)
    h,w = left_image.shape

    window_size_half = int(window_size/2)
    disparity_left =np.zeros((h,w))
    #breakpoint()
    for i in range(window_size_half,h - window_size_half):
        #l = [0]*left_image.shape[1]
        for j in range(window_size_half,w - window_size_half):


            min_distance = 65535
            min_j  = 0

            for disparity in range(max_disparity):
                distance = 0
                temp =0;
                for l in range(-window_size_half, window_size_half):
                    for m in range(-window_size_half, window_size_half):
                        if(direction == 0):
                            temp= int(left[i+l, j+m]) -  int(right[i+l, (j+m)-disparity])
                        else:
                            temp= int(right[i+l, j+m]) -  int(left[i+l, (j+m+disparity)%w])
                        
                            
                        distance += temp*temp
                if (distance <min_distance):
                    min_distance=distance
                    min_j = disparity
           
            disparity_left[i,j] = min_j

    e2 = cv.getTickCount()
    print("time taken is ", (e2-e1)/cv.getTickFrequency())
   
    return disparity_left


def add_contrast(image):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control
    # Initialize values
    print(' Basic Linear Transforms ')
    print('-------------------------')
    try:
        alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
        beta = int(input('* Enter the beta value [0-100]: '))
    except ValueError:
        print('Error, not a number')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
    return new_image



def calculate_SSD_over_range(left_gray, right_gray):
    left_gray_noise =  add_gaussian_noise(0,15, left_gray)
    right_gray_noise = add_gaussian_noise(0,15, right_gray)
    for i in range(4,17,4):
        print("Calculate disparity image with noise for window size {}".format(i))
        disparity_left = calculate_SSD(left_gray_noise,right_gray_noise,0,i,40)
        disparity_right = calculate_SSD(left_gray_noise,right_gray_noise,1,i,40)
        disparity_left = ps.threshold_image(disparity_left,40)
        disparity_right = ps.threshold_image(disparity_right,40)
        #ps.display_image("disparity image", disparity_left)
        file_name_left = "disparity_image_left" + "window_size" + str(i)+ "noise"
        file_name_right = "disparity_image_right" + "window_size" + str(i)+"noise"
        
        ps.save_image(file_name_left, disparity_left)
        ps.save_image(file_name_right, disparity_right)
    


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

    
    left_gray = cv.cvtColor(left_image,cv.COLOR_BGR2GRAY)
   
    right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    calculate_SSD_over_range(left_gray, right_gray)

    

   

    






# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
