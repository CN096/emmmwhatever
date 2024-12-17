import numpy as np
import cv2

def read_3d_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)

# 读取p3d文件
points_3d = read_3d_points('house.p3d')
# points_3d=np.array([
#     [100, 100, 100],  # 三维坐标
#     [200, 200, 200]
# ])
#print(points_3d)
# 内参数矩阵 (假设焦距为 800 像素，图像大小为 1280x960)
focal_length = 800
principal_point = np.array([640, 480])
K_intrinsic = np.array([[focal_length, 0, principal_point[0]],
                        [0, focal_length, principal_point[1]],
                        [0, 0, 1]])

# 外参数矩阵 (假设没有旋转，简单位移)
#R_extrinsic = np.eye(3)  # 单位矩阵代表没有旋转
R_extrinsic = np.array([[0.70710678, 0.70710678, 0],
                        [-0.70710678, 0.70710678, 0],
                        [0, 0, 1]])


T_extrinsic = np.array([0, 0, 10])  # 位移10个单位沿z轴

# 输出设定的内外参数
print("K_intrinsic:\n", K_intrinsic)
print("R_extrinsic:\n", R_extrinsic)
print("T_extrinsic:\n", T_extrinsic)
def project_points(points_3d, K, R, T):
    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # 转为齐次坐标
    print(points_3d_homo)
    RT = np.hstack((R, T.reshape(-1, 1)))  # 拼接 [R|T] 矩阵
    points_camera = RT.dot(points_3d_homo.T).T  # 应用外参矩阵
    points_2d_homo = K.dot(points_camera.T).T  # 应用内参矩阵
    print("points_2d_homo:",points_2d_homo)
    
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2].reshape(-1, 1)  # 转换为非齐次坐标
    return points_2d

# 投影三维点到二维图像平面
points_2d = project_points(points_3d, K_intrinsic, R_extrinsic, T_extrinsic)

# 输出投影后的像素坐标
print("Projected 2D points:\n", points_2d)
def decompose_projection_matrix(P):
    K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    
    # 调整内参矩阵 K 的形状
    K = K[:3, :3]
    
    # 调整外参矩阵 [R | t] 的形状
    print(t)
    R = R[:3, :3]
    t = t[:3]/-t[-1]
    t = t[:3, 0]
    
    return K, R, t
def decompose_projection_matrix1(P):
    temp = np.linalg.inv(P[0:3, 0:3])
    R, K = np.linalg.qr(temp)
    t = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(t) < 0:
        t[1,1] *= -1
    #R = np.linalg.inv(R)
    R = np.dot(t,R).T # T 的逆矩阵为其自身
    K = np.linalg.inv(K)
    K = K / K[2, 2]
    K = abs(K)  # need abs
    T =  np.matmul(temp, P[:, 3])
    return K, R, T
def estimate_camera_parameters(points_3d, points_2d):
    num_points = points_3d.shape[0]
    A = []
    
    for i in range(num_points):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    
    # 通过 SVD 分解来求解 P 矩阵
    _, _, Vt = np.linalg.svd(A)
    #P = Vt[-1].reshape(3, 4)
    P=Vt[-1,:]
    P=P/P[-1]
    P=P.reshape(3,4)
    
    # 从 P 矩阵中分解出 K, R, T
    # M = P[:, :3]  # 前 3 列是内外参数的乘积
    # R, K = np.linalg.qr(np.linalg.inv(M))  # QR 分解求 R 和 K
    
    # T = np.linalg.inv(K).dot(P[:, 3])  # 位移向量 T


    # R,K = np.linalg.qr(P[:,:3])
    # T = np.diag(np.sign(np.diag(K)))
    # if np.linalg.det(T) < 0:
    #     T[1,1] *= -1
    # K1 = np.dot(K,T)/K[2,2]
    # R = np.dot(T,R) # T 的逆矩阵为其自身
    # t = np.dot(np.linalg.inv(K),P[:,3])
    
    K, R, t = decompose_projection_matrix1(P)

    #K,R,t=decompose_projection_matrix(P)
    #K=K/K[-1,2]

    
    return K, R, t

# 估计相机参数
K_estimated, R_estimated, T_estimated = estimate_camera_parameters(points_3d, points_2d)

# 输出估计的参数
print("Estimated K:\n", K_estimated)
print("Estimated R:\n", R_estimated)
print("Estimated T:\n", T_estimated)
