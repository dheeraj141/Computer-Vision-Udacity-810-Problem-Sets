import numpy as np
import cv2 as cv

import sys

def display_image( description, img):
    cv.imshow(description, img)
    cv.waitKey(0)

def construct_diagonal_matrix(s):
    D = np.zeros((len(s), len(s)), dtype=np.float64)
    for i in range(len(s)):
        D[i,i] = s[i]
    print(D)
    return D


def build_matrix(image_points1, image_points2):
    A = np.zeros((len(image_points2), 9), dtype=np.float64)
    temp = np.zeros((3), dtype=np.float64)
    temp1 = np.zeros((9), dtype=np.float64)
    h,w =  image_points2.shape
    #breakpoint()
    # left  = image points 2 or b and right is image points 1 or a
    for i in range(h):
        temp[0] = image_points2[i][0]
        temp[1] = image_points2[i][1]
        temp[2] = 1
        x = image_points1[i][0]
        y = image_points1[i][1]
        for j in range(0,9,3):
            temp1[j:j+3] = temp
        temp1[0:3]*=x
        temp1[3:6]*=y
        A[i] =  temp1
    return A;



def F_matrix_using_SVD(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    v = vh.transpose()
    #print(v)
    f = v[:,8]
    f = f.reshape((3,3))
    #found the f matrix  and now doing svd on f
    u,s,vh = np.linalg.svd(f, full_matrices=True)
    #breakpoint()
    s[len(s) -1] = 0
    #reconstructing F matrix
    D = construct_diagonal_matrix(s)
    F = np.matmul(u,np.matmul(D,vh))
    return F


def F_matrix_using_LST(A):
    h,w = A.shape
    A = A[0:h, 0:w-1]
    B = np.ones((20,1), dtype= np.float64)
    B*=-1
    X = np.linalg.lstsq(A, B, rcond=None)[0]
    F = np.ones((9,1), dtype= np.float64)
    F[0:8] = X
    F = F.reshape((3,3))
    return F





def cross_product(line1,line2):
    x = np.zeros((3,3), dtype=np.float64)
    x[0,1] = -line1[2]
    x[0,2] = line1[1]
    x[1,0] = line1[2]
    x[1,2] = -line1[0]
    x[2,0] = -line1[1]
    x[2,1] = line1[0]
    #print(x)
    p2 = np.zeros((3), dtype=np.float64)
    p2[0] = line2[0]
    p2[1] = line2[1]
    p2[2] = line2[2]
    #print(p2)

    p2 = p2.transpose()
    #print(p2)
    #print(p2.shape)
    y = np.matmul(x,p2)
    return y


def transformation_matrix (image_points):
    image_points_transformed = np.copy(image_points)
    # maximum absolute value
    h,w = image_points.shape
    max = -sys.maxsize -1
    for i in range (h):
        temp1 = np.absolute(image_points[i][ 0])
        temp2 = np.absolute(image_points[i][ 1])
        if(max < temp1):
            max = temp1
        if(max <temp2):
            max = temp2

    x_mean = 0
    y_mean = 0
    for i in range(h):
        x_mean +=image_points[i][0]
        y_mean+=image_points[i][1]
    x_mean /=h
    y_mean/=h
    s = 1/max
    scale_matrix  = np.array([[s, 0, 0],[ 0, s, 0], [0,0, 1]], dtype=np.float64)
    offset_matrix  = np.array([[1, 0, -x_mean],[ 0, 1, -y_mean], [0,0, 1]], dtype=np.float64)
    transformation = np.matmul(scale_matrix, offset_matrix)
    for i in range(h):
         temp = np.array([[image_points[i][0]], [image_points[i][1]], [1]], dtype=np.float64)
         temp1 = np.matmul(transformation, temp)
         image_points_transformed[i][0] = temp1[0]/temp1[2]
         image_points_transformed[i][1] = temp1[1]/temp1[2]
    return image_points_transformed






def extract_points(image, world):

    x = image.split('\n')
    y =  world.split('\n')
    image_points = np.zeros((len(x)-1,2), dtype =np.float64)
    for i in range(0,len(x) -1):
        l = x[i].split('  ')
        image_points[i,0] = np.float64(l[0])
        image_points[i,1] = np.float64(l[1])

    #print(image_points)
    y = world.split('\n')
    #breakpoint()
    world_points = np.zeros((len(y) -1,2), dtype=np.float64)
    for i in range(0,len(y) -1):
        l = y[i].split('  ')
        world_points[i,0] = np.float64(l[0])
        world_points[i,1] = np.float64(l[1])
        #world_points[i,2] = np.np.float64(l[2])

    #print(world_points)
    return image_points, world_points





def draw_epipolar_lines(image_points, f_matrix, line1, line2, image):
    # draw lines in image a and subsitute points of image b
    h,w = image_points.shape
    temp = np.zeros((3), dtype=np.float64)
    for i in range(h):
        temp [0] = image_points[i][0]
        temp[1] = image_points[i][1]
        temp[2] = 1
        #line in image b


        line_b = np.matmul(f_matrix, temp)
        point1 = cross_product(line_b, line1)
        point2 = cross_product(line_b, line2)

        #breakpoint()

        x1 = int(point1[0]/point1[2])
        y1 =int(point1[1]/point1[2])
        x2 =int(point2[0]/point2[2])
        y2 =int( point2[1]/point2[2])
        cv.line(image, (x1,y1),(x2,y2), (255,0,0), 1)
    display_image("epipolar lines", image)








def main(argv):
    #fp1 = open("../pts2d-pic_a.txt", 'r')
    fp1 = open("../pts2d-pic_a.txt", 'r')

    if fp1 is None:
        print("file can't be opened\n")
    #fp2 = open("../pts3d.txt", 'r')
    fp2 = open("../pts2d-pic_b.txt", 'r')

    if fp2 is None:
        print("file can't be opened\n")



    image_points = fp1.read()
    world_points = fp2.read()
    #print(image_points,'\n')
    #print(world_points,'\n')
    image_points_a, image_points_b = extract_points(image_points, world_points)
    print(image_points_a,'\n')
    print(image_points_b,'\n')
    breakpoint()




    A = build_matrix(image_points_a, image_points_b)
    F_SVD = F_matrix_using_SVD(A)

    F_lst = F_matrix_using_LST(A)

    #F = np.matmul(T_b.transpose(),np.matmul(F, T_a))
    breakpoint();
    #print(F)
    point_1 = np.zeros((3), dtype=np.float64)
    point_2 = np.zeros((3), dtype=np.float64)
    image_a = cv.imread('../pic_a.jpg',cv.IMREAD_COLOR)
    image_b = cv.imread('../pic_b.jpg', cv.IMREAD_COLOR)
    display_image('image 1', image_a)
    display_image('image 2', image_b)


    h,w,c  = image_a.shape
    h1,w1,c1 = image_b.shape
    #cv.line(image_a,(0,0),(0,h),(255,0,0),2)
    point_1[0] = 0
    point_1[1] = 0
    point_1[2] =1
    point_2[0] = 0
    point_2[1] = h
    point_2[2] =1
    line_1 = cross_product(point_1,point_2)
    point_1[0] = w
    point_1[1] = 0
    point_1[2] =1
    point_2[0] = w
    point_2[1] = h
    point_2[2] =1
    #breakpoint()
    # left  = image points b and right is image points a
    # put points of image b in F matrix and draw line in image A
    line_2 = cross_product(point_1, point_2)
    #drawing epipolar lines in image a
    draw_epipolar_lines(image_points_b, F_SVD, line_1, line_2 , image_a)

    #drawing epipolar lines in image b
    draw_epipolar_lines(image_points_a,  F_SVD.transpose(), line_1, line_2 , image_b)

    # f matrix using normalized points
    breakpoint()

    T_a = transformation_matrix(image_points_a)
    T_b = transformation_matrix(image_points_b)

    # building the matrix
    A = build_matrix(T_a, T_b)
    # fmatrix
    F_SVD_normalized = F_matrix_using_SVD(A)

    #drawing epipolar lines in image a
    draw_epipolar_lines(image_points_b, F_SVD_normalized, line_1, line_2 , image_a)

    #drawing epipolar lines in image b
    draw_epipolar_lines(image_points_a,  F_SVD_normalized.transpose(), line_1, line_2 , image_b)














# if someone import this module then this line makes sure that it does not Run
#ion its own
if __name__ == "__main__":
    main(sys.argv[1:])
