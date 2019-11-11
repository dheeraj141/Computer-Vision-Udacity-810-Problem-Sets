import numpy as np
import cv2 as cv
import sys
import ps12 as ps
import sys
import reference_KD_tree as kd




def matching_descriptors(kp1, des1,kp2, des2):
    size =  len(kp1); matching = np.ones((size));

    for i in range(len(kp1)):
        print("matching for kp {}".format(i))
        temp1 = np.copy(des1[i])
        L2_norm = sys.maxsize
        temp2 = np.copy(des2)
        temp2-=temp1;
        temp2**=2;
        temp2 = np.sqrt(temp2)
        sum =[np.sum(x) for x in temp2]
        index = np.where(sum == np.amin(sum))
        matching[i] = index[0][0]

    return matching

def calculate_descriptors(img1):
    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()
    grad_x = gradient(img1, 0)
    grad_y = gradient(img1,1)
    alpha = 0.04
    corners = harris_Corners(grad_x, grad_y,0, 5,0.04)
    #breakpoint()
    #ANMS(corners)
    #corners = cv.dilate(corners, None)
    #img1[corners>0.4*corners.max()]=[0,0,255]
    #ps.display_image("corners", img1)
    #breakpoint()

    corner_ptr = thresholding_and_non_maximum_supression(corners, img1)
    #y,x = np.nonzero(corners >0.5*corners.max())

    angle = np.arctan2(grad_y, grad_x)
    #draw_line(angle,corner_ptr, img1)
    kp = [cv.KeyPoint(x[1], x[0], 1, angle[int(x[0]), int(x[1])]) for x in corner_ptr]
    kp, des = sift.compute(gray, kp)
    return corner_ptr,kp,des




def feature_matching_using_KD(kp1, des1, kp2, des2):
    size =  len(kp1); matching = np.ones((size));
    matching  = np.ones((len(kp1)))
    #building the KD tree using the feature points of image 2

    #tree =  kd.create(dimensions = 128)
    tree = kd.create(dimensions =  128)


    for i in range(len(kp1)):
        tree.add(des2[i],i)
    for i in range(size):
        x = tree.search_knn(des1[i], 2)
        matching[i] = x[0][0].return_index()
    return matching















def draw_keypoints(corner1,corner2, matching, img3, w1):

    for i in range(25):
        index = np.random.randint(0, len(corner1))
        x1 =int(corner1[index][1]); y1 = int(corner1[index][0]);
        k = int(matching[index])
        x2 =int(corner2[k][1] + w1); y2 = int(corner2[k][0]);
        #breakpoint()
        color1 = (list(np.random.choice(range(256), size=3)))
        color =[int(color1[0]), int(color1[1]), int(color1[2])]
        cv.circle(img3, (x1,y1), 2,color,2)
        cv.circle(img3, (x2, y2), 2,color,2)
        cv.line(img3, (x1,y1), (x2, y2), color, 2)
    ps.display_image("matched", img3)






def main(argv):

    if(len(argv) < 1):
        print("not enough parameters\n")
        print("usage PS1-1.py <path to image>\n")
        return -1


    img1 = cv.imread(argv[0],cv.IMREAD_COLOR)
    if img1 is None:
        print("Error opening image\n")
        return -1
    img2 = cv.imread(argv[1],cv.IMREAD_COLOR)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    h1,w1 = gray1.shape; h2, w2 = gray2.shape;
    breakpoint()
    img3 = np.concatenate((img1, img2), axis = 1)
    #ps.display_image("concatenated image", img3)
    corner1, kp1,des1 = calculate_descriptors(img1)
    corner2, kp2,des2 = calculate_descriptors(img2)
    matching = feature_matching_using_KD(kp1, des1, kp2, des2)
    draw_keypoints(corner1, corner2, matching, img3, w1)
    matching = matching_descriptors(kp1, des1, kp2, des2)
    img4 = np.copy(img3)
    draw_keypoints(corner1, corner2, matching, img4, w1)
    #ransac(corner1, corner2, matching, gray1,gray2)






    breakpoint()


    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    #Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

    #cv.drawMatchesKnn expects list of lists as matches.
    breakpoint()
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ps.display_image("matching", img3)
















# if someone import this module then this line makes sure that it does not Run
# on its own
if __name__ == "__main__":
    main(sys.argv[1:])
