def non_maximum_supression(corners, img):

    """ input is the harris corners image
     Non maximum suppression code to select spatially well distributed
    keypoints and here the algorithm is simple and easy to understand
    """

    harris = np.asarray(corners)
    threshold = 0.1*np.amax(corners)
    max_overlap =8; max_overlap**2; window = 8; offset =  window//2

    h,w = corners.shape

    corner_ptr = []
    pick = []
    for m in range(offset , h-offset, offset):
        for n in range(offset, w-offset, offset):
            #window extracted
            temp =  harris[m-offset:m+offset, n-offset:n+offset]
            a,b  = np.nonzero(temp > threshold )
            if (len(a) >0):
                temp3 = temp[a,b]
                list1 = np.ones((len(a), 3))

                supress = []
                list1[:, 0] = a; list1[:,1] = b; list1[:, 2] =temp3;

                ind = np.argsort( list1[:,2] ); list1 = list1[ind]
                list1[:,2] = list1[:,2]//list1[0,2];

                while(len(list1) > 0):
                    last = len(list1) - 1;
                    cor = np.copy(list1[last]);
                    i = list1[last]; cor[0]+=(m-offset); cor[1]+=(n-offset);
                    pick.append(cor)
                    supress= [last]
                    temp_list = np.copy(list1[0:last])
                    temp_list[:,0]-=i[0]
                    temp_list[:,0]**=2
                    temp_list[:,1]-=i[1]
                    temp_list[:,1]**=2
                    list2 = np.copy(temp_list[:,0]+ temp_list[:, 1])
                    pos = np.nonzero(list2 <= max_overlap)
                    list1 = np.delete(list1, supress, 0)

                    list1 = np.delete(list1,pos, 0);

    key_points = np.ones((len(pick), 3))
    for i in range(len(pick)):
        key_points[i] = pick[i]
    ind = np.argsort( key_points[:,2] ); key_points = key_points[ind]
    return key_points[0:500]
