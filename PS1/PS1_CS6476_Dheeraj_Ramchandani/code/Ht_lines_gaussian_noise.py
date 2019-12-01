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





    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_smooth = cv.GaussianBlur(gray,(5,5),4,4)
    edges_original = cv.Canny(gray,140,210)
    ps.display_image("edges in the original image", edges_original)
    ps.save_image("ps1-3-a",gray_smooth)
    ps.save_image("ps1-3-b-1", edges_original)


    edges_smoothed = cv.Canny(gray_smooth,140,210)
    ps.display_image("edges in the smoothened image", edges_smoothed)
    ps.save_image("ps1-3-b-2", edges_smoothed)

    hough_space = ps.hough_transform(edges_smoothed,gray_smooth)

    #display_image("Displaying 3D plot of the hough space",hough_space)
    ps.plot_3d_graphs(hough_space, "hough_space_ps1-3")
    smooth = ps.detect_line_in_image(hough_space,img,0.5)
    ps.display_image("PS3", img)
    ps.save_image("ps1-3-c",img)

# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
