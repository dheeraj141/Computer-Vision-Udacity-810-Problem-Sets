import numpy as np
import cv2 as cv
import ps12 as ps
import sys

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
    monochrome_image = img[:,:,1]
    ps.display_image("monochrome image", monochrome_image)

    monochrome_smooth_image = cv.GaussianBlur(monochrome_image,(5,5),4,4)
    ps.display_image("smoothened  monochrome image", monochrome_smooth_image)
    ps.save_image("ps1-4-a", monochrome_smooth_image)


    edges_smooth_monochrome = cv.Canny(monochrome_smooth_image,100,210)
    ps.display_image("edges in the smoothed monochrome image", edges_smooth_monochrome)
    ps.save_image("ps1-4-b", edges_smooth_monochrome)

    hough_space = ps.hough_transform(edges_smooth_monochrome, monochrome_smooth_image)
    ps.plot_3d_graphs(hough_space,"hough_space_ps1-4-c")
    #measuring time to detect lines in the image

    img1 = ps.detect_line_in_image(hough_space,monochrome_smooth_image,0.5)

    ps.display_image("Lines in the image are",img1)

    ps.save_image("ps1-4-c-1", img1)



# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
