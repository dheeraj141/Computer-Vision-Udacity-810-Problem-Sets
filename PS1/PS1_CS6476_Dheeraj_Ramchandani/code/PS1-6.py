import numpy as np
import cv2 as cv
import ps12 as ps
import sys
import math

def condition_check(x,y,max_x,max_y):
    if(x >=0 and x <max_x):
        if(y >=0 and y<max_y):
            return 1
        else :
            return 0
    else :
        return 0

#function to detect lines in the hough space
#input  :     image, threshold (above which the elements will be considered)
#output :     image with lines drawn on it


def detect_line_in_image(edges, hough_space,img,threshold):

    image_size_x = img.shape[0]
    image_size_y=img.shape[1]
    maximum_distance = int(np.sqrt(image_size_x*image_size_x + image_size_y*image_size_y))
    offset=maximum_distance
    #extracting the maximum values form the hough space by seeing the
    limit = (int)(threshold*255)
    print(limit)
    for d in range(hough_space.shape[0]):
        for angle in range(hough_space.shape[1]):
            if(hough_space[d,angle] >limit):
                if (angle ==0):
                    #print("got zero")
                    y_value = d-offset
                    cv.line(img,(0,y_value), (img.shape[1],y_value),(0,255,0),2)
                    continue
                elif(angle >85 and angle <95):
                    #print("got 90")
                    #x_value=d-offset
                    #cv.line(img, (x_value,0), (x_value,img.shape[0]),(0,255,0),2)
                    continue
                else:

                    theta = np.deg2rad(angle)
                    r = d-offset
                    for x_value in range(0,img.shape[1],2):
                        x1 =x_value
                        x2 = x_value+1
                        y1 = int((r - x1*np.cos(theta))/np.sin(theta))
                        y2 = int((r - x2*np.cos(theta))/np.sin(theta))
                        #breakpoint()
                        if(condition_check(x1,y1,edges.shape[1],edges.shape[0])>0 and condition_check(x2,y2,edges.shape[1],edges.shape[0]) >0):
                            if(edges[y1,x1] and edges[y2,x2]):
                                cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    return img



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
    print(type(img))
    print(img.size,img.ndim, img.shape)
    smooth_image = cv.GaussianBlur(img,(5,5),4,4)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ps.display_image("gray image ",gray)
    gray_smooth_image = cv.GaussianBlur(gray,(5,5),4,4)
    ps.display_image(" gray smoothened  image", gray_smooth_image)


    lower_threshold = int(input("enter the lower threshold value for canny\n"))
    upper_threshold = int(input("enter the upper threshold value for canny\n"))
    edges_smooth = cv.Canny(gray_smooth_image,lower_threshold,upper_threshold)
    ps.display_image("edges in the smoothed monochrome image", edges_smooth)
    e1 = cv.getTickCount()
    hough_space = ps.hough_transform(edges_smooth, gray_smooth_image)
    #ps.plot_3d_graphs(hough_space)
    #measuring time to detect lines in the image
    threshold = float(input("enter the hough accululator threshold value\n"))

    #img1 = ps.detect_line_in_image(edges_smooth,hough_space,smooth_image,threshold)
    img1 = ps.detect_line_in_image(hough_space,smooth_image,threshold)
    e2 = cv.getTickCount()
    time_taken = (e2 -e1)/cv.getTickFrequency()
    print("The time taken to calculate the lines in the image is {} seconds\n".format(time_taken))
    ps.display_image("Lines in the image are",img1)
    ps.save_image("ps1-6-a", img1)




# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
