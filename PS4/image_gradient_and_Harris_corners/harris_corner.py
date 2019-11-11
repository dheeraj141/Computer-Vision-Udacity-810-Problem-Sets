import numpy as np
import cv2 as cv
import sys




def draw_corners(corner, img):
    y, x = np.nonzero(corner > 0.4*corner.max())
    for i in range(len(x)):
        x1 = int(x[i]); y1 = int(y[i])
        color1 = (list(np.random.choice(range(256), size=3)))
        color =[int(color1[0]), int(color1[1]), int(color1[2])]
        cv.circle(img, (x1,y1), 4,color,4)


    display_image("image 1", img)



def display_image( description, img):
    cv.imshow(description, img)
    cv.waitKey(0)




#function to calculate the harris corners
#input      img and the window size
#output     corners image

def harris_Corners(img, window_size):
    """ Harris corners function takes the image and builds the R corner
    response image"""
    smooth = cv.GaussianBlur(img,(3,3),0)
    gray = cv.cvtColor(smooth, cv.COLOR_BGR2GRAY)
    grad_x =  cv.Sobel(gray,-1,1,0,ksize = 3)
    grad_y =  cv.Sobel(gray,-1,0,1,ksize = 3)

    h,w = grad_x.shape
    corners = np.zeros((h,w), dtype = np.float64)
    grad_x = grad_x.astype(np.float64)
    grad_y = grad_y.astype(np.float64)

    grad_xx = grad_x**2
    grad_xx = cv.GaussianBlur(grad_xx,(3,3),0.5)
    grad_yy = grad_y**2
    grad_yy = cv.GaussianBlur(grad_yy,(3,3),0.5)
    grad_xy = grad_x *grad_y
    grad_xy = cv.GaussianBlur(grad_xy, (3,3), 0.5)



    offset = int(window_size/2)
    for i in range(offset, h-offset):
        for j in range(offset, w-offset):
            temp1 = grad_xx[i-offset:i+offset+1, j-offset:j+offset+1]
            temp2 = grad_yy[i-offset:i+offset+1, j-offset:j+offset+1]
            temp3 = grad_xy[i-offset:i+offset+1, j-offset:j+offset+1]
            a = np.sum(temp1)
            b = np.sum(temp2)
            c= np.sum(temp3)
            det  = a*b - (c*c)
            trace = a+b
            R = det - 0.04*(trace**2)
            corners[i,j] = R

    return corners






def main(argv):

    if(len(argv) < 1):
        print("not enough parameters\n")
        print("usage PS1-1.py <path to image>\n")
        return -1


    img1 = cv.imread(argv[0],cv.IMREAD_COLOR)
    if img1 is None:
        print("Error opening image\n")
        return -1
    corners = harris_Corners(img1, 5)
    draw_corners(corners, img1)







# if someone import this module then this line makes sure that it does not Run
# on its own
if __name__ == "__main__":
    main(sys.argv[1:])
