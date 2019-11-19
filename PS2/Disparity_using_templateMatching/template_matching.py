import cv2 as cv
import numpy as np
import ps12 as ps
import sys
import math



def check_img(img):
    if img is None:
        return 0
    else:
        return 1
def make_template( y,  x, kernel, left_image):
    template = np.zeros((kernel, kernel))
    kernel_half = int(kernel/2)

    for i in range(-kernel_half, kernel_half):
        for j in range(-kernel_half, kernel_half):
            template[y+i, x+j] =  left_image[y+i, x+j]
    ps.display_image("Template", template)
def add_gaussian_noise(mean, sd, left_image):
    h,w = left_image.shape
    noise = np.random.normal(mean, sd, (h,w))
    left = np.asarray(left_image)
    left = left.astype('float64')
    left+=noise
    left =left.astype('uint8')
    return left




def template_matching(left_image, right_image, direction, window_size):
    h,w = left_image.shape
    window_size_half = int(window_size/2)
    left = np.asarray(left_image)
    right = np.asarray(right_image)
    depth = np.zeros((h,w))

    for i in range(window_size_half, h-window_size_half):
        for j in range(window_size_half, w-window_size_half):
            #breakpoint()
            if (direction ==0):
                 template = left[i-window_size_half: i+ window_size_half, j-window_size_half: j+window_size_half]
                 image_patch = right[i-window_size_half: i+window_size_half, :]
            else :
                template = right[i-window_size_half: i+ window_size_half, j-window_size_half: j+window_size_half]
                image_patch = left[i-window_size_half: i+window_size_half, :]
            

            res = cv.matchTemplate(image_patch, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            #print(max_loc)
            depth[i,j] =int(math.fabs(j - (max_loc[0])))
    depth = depth.astype('uint8')
    ps.threshold_image(depth, np.amax(depth))
    return depth

    #ps.display_image("disparity image", depth)


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

    #ps.display_image("Image", left_image)
    left_gray = cv.cvtColor(left_image,cv.COLOR_BGR2GRAY)
    #ps.display_image("gray image ",left_gray)
    right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    #ps.display_image("graY image right", right_gray)
    left_gray_noise =  add_gaussian_noise(0,15, left_gray)
    right_gray_noise = add_gaussian_noise(0,15, right_gray)

    for i in range(6,9,2):
        print("Calculate disparity image with noise for window size {}".format(i))

        #calculating disparity
        
        disparity_left = template_matching(left_gray,right_gray,0,i)
        disparity_right = template_matching(left_gray,right_gray,1,i)

        #saving the images
        
        file_name_left = "disparity_image_left_template_matching" + str(i)
        file_name_right = "disparity_image_right_template_matching" + str(i)
        
        ps.save_image(file_name_left, disparity_left)
        ps.save_image(file_name_right, disparity_right)

        
        # calculating for noisy images
        
        disparity_left_noise = template_matching(left_gray_noise,right_gray_noise,0,i)
        disparity_right_noise = template_matching(left_gray_noise,right_gray_noise,1,i)
        
        #saving images

        
        file_name_left = file_name_left +"noise"
        file_name_right = file_name_right +"noise"
        
        ps.save_image(file_name_left, disparity_left_noise)
        ps.save_image(file_name_right, disparity_right_noise)
        
        

    








# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
