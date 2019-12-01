import numpy as np
import cv2 as cv
import ps12 as ps
import math
import sys


#main function to run the program
# input :       path to the image
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


    edges = cv.Canny(gray,100,200)
    ps.display_image("edges", edges)
    ps.save_image("ps1-1",edges)
# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
