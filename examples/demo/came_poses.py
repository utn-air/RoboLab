import numpy as np
from scipy.spatial.transform import Rotation as R


quat_xyzw = np.array([-0.393, -0.195, 0.399, 0.805])
r = R.from_quat(quat_xyzw, scalar_first=True)
print(r.as_matrix())
matrix = r.as_matrix()
# z0 = matrix[0, 2]
# z1 = matrix[1, 2]
# s = -2*z0*z1/(z0**2 + z1**2)
# c = 1 + z1*s/z0
# rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# rotated_mat = rot_mat @ matrix
# q = R.from_matrix(rotated_mat)
matrix[1, 0] *= -1
matrix[0, 1] *= -1
matrix[2, 1] *= -1
matrix[1, 2] *= -1
q = r.from_matrix(matrix)
print(q.as_quat(scalar_first=True))