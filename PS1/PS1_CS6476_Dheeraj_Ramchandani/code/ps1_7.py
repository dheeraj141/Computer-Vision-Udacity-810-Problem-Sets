import numpy as np
import cv2 as cv
import hough_transform_circle as cht
import ps12 as ps
import sys
import math

def main(argv):
    if(len(argv) < 1):
        print("not enough parameters\n")
        print("usage PS1-1.py <path to image>\n")
        return -1


    img = cv.imread(argv[0],cv.IMREAD_COLOR)
    if img is None:
        print("Error opening image\n")
        return -1
    ps.display_image("Image", img)
    smooth_image = cv.GaussianBlur(img,(5,5),4,4)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ps.display_image("gray image ",gray)
    gray_smooth_image = cv.GaussianBlur(gray,(5,5),4,4)
    ps.display_image(" gray smoothened  image", gray_smooth_image)


    edges_smooth = cv.Canny(gray_smooth_image,80,180)
    ps.display_image("edges in the smoothed monochrome image", edges_smooth)
    e1 = cv.getTickCount()
    x = cht.extract_edges(edges_smooth)
    rmin = int(input("enter the minimum raidus\n"))
    rmax = int(input("enter the max radius\n"))

    circles_in_image = cht.hough_space_circle(x,rmin,rmax)
    #centers = detecting_circle_using_chord(x)
    for j in circles_in_image:
        cv.circle(smooth_image,(j[0],j[1]), j[2], (0,255,0))
    ps.display_image("cirlces in image",smooth_image)
    e2 = cv.getTickCount()
    time_taken = (e2 -e1)/cv.getTickFrequency()
    print("The time taken to calculate the lines in the image is {} seconds\n".format(time_taken))
    ps.display_image("Lines in the image are",img1)




# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
