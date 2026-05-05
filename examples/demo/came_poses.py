import numpy as np
from scipy.spatial.transform import Rotation as R

# quat_xyzw = np.array([(0.842, 0.464, -0.112, -0.252)])
quat_xyzw = np.array([(0.86062666, 0.47192311, -0.07112259, -0.17762858)])
r = R.from_quat(quat_xyzw, scalar_first=True)
print(r.as_matrix())
matrix = r.as_matrix()

# rotate 45 degrees around Z axis
axis = [1, 0, 0]  # Z-axis
angle = np.radians(5)  # 45 degrees
rotation = R.from_rotvec(angle * np.array(axis))

# rotate camera rotation by 45 degrees around Z axis
rotated_matrix = rotation.as_matrix() @ matrix
print(rotated_matrix)
q = R.from_matrix(rotated_matrix)
print(q.as_quat(scalar_first=True))
exit()


# z0 = matrix[0, 2]
# z1 = matrix[1, 2]
# s = -2*z0*z1/(z0**2 + z1**2)
# c = 1 + z1*s/z0
# rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# rotated_mat = rot_mat @ matrix
# q = R.from_matrix(rotated_mat)
# matrix[1, 0] *= -1
# matrix[0, 1] *= -1
# matrix[2, 1] *= -1
# matrix[1, 2] *= -1
# q = r.from_matrix(matrix)
# print(q.as_quat(scalar_first=True))