import numpy as np
import sys





def build_mask(l, a):
    for i in range(8,12):
        l[i] = -a

    return l




def check_code(world_point, m_matrix):
    p = np.zeros((4), dtype=np.float64)
    p[0] = 1.2323
    p[1] = 1.4421
    p[2] = 0.4506
    p[3] = 1
    q = np.transpose(p)
    n = np.matmul(m_matrix,q)
    return n
def calculate_error(image_points, index, n):
    u = n[0]/n[2]
    v = n[1]/n[2]

    error =0
    x = image_points[index][0]
    y = image_points[index][1]
    //print("predicted points ", u, v , "actual output", x, y)
    error= pow((u-x),2) + pow((v-y),2)
    return np.sqrt(error)


def calculate_residue(m_matrix, image_points, world_points):
    p = np.zeros((4), dtype= np.float64)
    error = 0
    for i in range(4):
        for j in range(3):
            p[j] = world_points[i+16][j]
        p[3] = 1
        q = np.transpose(p)
        n = np.matmul(m_matrix,q)
        error+=calculate_error(image_points, i+16, n)
    return error






def generate_random_points( k):
    points = np.empty((k), dtype=np.int32)
    if (k ==16):
        for i in range(k):
            points[i] = i
    else:
        count = 0
        i = 0
        while( i <k*2 and count <k):
            x = np.random.randint(0,16)
            if x in points:
                continue
            else :
                points[count] = x
                count+=1
            i+=1
        #print(points)

    return points



def generate_A_matrix(image_points, world_points, points):
    A = np.zeros((2*points.size, 12), dtype=np.float64)
    mask_A = np.zeros((12), dtype = np.float64)
    mask_B = np.zeros((12), dtype = np.float64)
    mask_A = [1,1,1,1,0,0,0,0,1,1,1,1]
    mask_B = [0,0,0,0,1,1,1,1,1,1,1,1]
    temp = np.zeros((12), dtype =np.float64)
    temp1 = np.zeros((4), dtype=np.float64)
    count = 0
    for i in points:
        #breakpoint()
        mask_A = build_mask(mask_A, image_points[i][0])
        mask_B = build_mask(mask_B, image_points[i][1])
        #breakpoint()
        for j in range(3):
            temp1[j] = world_points[i][j]
        temp1[3] = 1
        temp[0:4] = temp1
        temp[4:8] = temp1
        temp[8:12] = temp1
        print(temp)
        A[count] = temp
        A[count+1]  =  temp



        A[count]*= mask_A
        A[count+1] *=mask_B
        count+=2
    B = np.ones((2*points.size,1), dtype = np.float64)
    count = 0
    for i in points:
        B[count] = image_points[i][0]
        B[count+1] = image_points[i][1]
        count+=2
    return A, B


def M_matrix_SVD(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v = vh.transpose()
    m = v[:,11]
    #print (m)
    m = m.reshape((3,4))
    #print(m.shape)

    return m



def M_matrix_LST(A, image_points):
    h,w = A.shape
    A = A[0:h, 0:w-1]

    X = np.linalg.lstsq(A, B, rcond=None)[0]
    M = np.ones((12,1), dtype= np.float64)
    M[0:11] = X
    M = M.reshape((3,4))
    return M










def extract_points(image, world):

    x = image.split('\n')
    y =  world.split('\n')
    image_points = np.zeros((len(x)-1,2), dtype =np.float64)
    for i in range(0,len(x) -1):
        l = x[i].split('  ')
        image_points[i,0] = np.float64(l[0])
        image_points[i,1] = np.float64(l[1])
    y = world.split('\n')
    world_points = np.zeros((len(y) -1,3), dtype=np.float64)
    for i in range(0,len(y) -1):
        l = y[i].split(' ')
        world_points[i,0] = np.float64(l[0])
        world_points[i,1] = np.float64(l[1])
        world_points[i,2] = np.float64(l[2])
    return image_points, world_points



def find_centre(m_matrix):
    Q = m_matrix[:,:-1]
    m4 = m_matrix[:,-1]
    a = np.linalg.inv(Q)
    c = -np.matmul(a,m4)
    return c














fp1 = open("../pts2d-pic_b.txt", 'r')
#fp1 = open("../pts2d-norm-pic_a.txt", 'r')

if fp1 is None:
    print("file can't be opened\n")
fp2 = open("../pts3d.txt", 'r')
#fp2 = open("../pts3d-norm.txt", 'r')

if fp2 is None:
    print("file can't be opened\n")



image = fp1.read()
world = fp2.read()
breakpoint()
image_points, world_points =extract_points(image, world)
points = generate_random_points(16)
A,B = generate_A_matrix(image_points, world_points, points)


M_LST  = M_matrix_LST(A, B)
M_SVD = M_matrix_SVD(A)
M_MIN = M_SVD.copy()
image_point = check_code(world_points, M_LST)
x = image_point[0]/image_point[2]
y = image_point[1]/image_point[2]
error =  pow((x -0.1419),2) + pow((y +0.4518),2)
error = np.sqrt(error)
k = [8,12,16]
total_error = [0,0,0]
count = 0
min_error  =sys.maxsize
index = 0
for j in k:
    error = 0
    for i in range(10):
        points = generate_random_points(j)
        A,B = generate_A_matrix(image_points, world_points, points)
        M_SVD = M_matrix_SVD(A)
        error1 = calculate_residue(M_SVD, image_points, world_points)
        error+=error1
        if(min_error >error1):
            index =j
            min_error = error1
            M_MIN = M_SVD
    total_error[count] = error
    count+=1

print(M_MIN)
c = find_centre(M_MIN)
