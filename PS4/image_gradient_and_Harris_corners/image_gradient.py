import numpy as np
import cv2 as cv
import sys
import ps12 as ps


def main(argv):

    if(len(argv) < 1):
        print("not enough parameters\n")
        print("usage PS1-1.py <path to image>\n")
        return -1


    img = cv.imread(argv[0],cv.IMREAD_COLOR)
    if img is None:
        print("Error opening image\n")
        return -1
    smooth = cv.GaussianBlur(img,(3,3),0)
    gray = cv.cvtColor(smooth, cv.COLOR_BGR2GRAY)
    grad_x =  cv.Sobel(gray,-1,1,0,ksize = 3)

    grad_y =  cv.Sobel(gray,-1,0,1,ksize = 3)
    img3 = np.concatenate((grad_x, grad_y), axis = 1)
    img3[img3>0] = 255
    img3[img3 == 0] = 128
    img3[img3 <0] = 0
    ps.display_image("concatenated image", img3)












# if someone import this module then this line makes sure that it does not Run
# on its own
if __name__ == "__main__":
    main(sys.argv[1:])
