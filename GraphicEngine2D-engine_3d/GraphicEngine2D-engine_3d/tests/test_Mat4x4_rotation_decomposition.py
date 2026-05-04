import numpy as np

from src.math.Mat4x4 import Mat4x4
from src.math.Quaternion import Quaternion
from src.math.Vec3 import Vec3
from src.math.utils_matrix import is_same_matrix
from src.math.utils_quat import rotation_matrix_to_quaternion

M_orig = Mat4x4(
    [[0.87559502, -0.38175263, 0.29597008, 0.],
     [0.42003109, 0.90430386, -0.07621294, 0.],
     [-0.2385524, 0.19104831, 0.95215193, 0.],
     [0., 0., 0., 1.]]
)

q_orig = Quaternion(0.96592583, 0.0691723, 0.1383446, 0.2075169)

axis_orig = Vec3([0.26726124, 0.53452248, 0.80178373])

angle_orig = np.radians(30)


def test_rotation_from_quaternion_and_directly():
    angle, axis = np.radians(30), Vec3(1, 2, 3).normalized()

    m_rot = Mat4x4.rotation(angle, axis)
    assert is_same_matrix(m_rot, M_orig), "Transformation matrices are not equal"

    q = Quaternion.rotation(angle, axis)
    m_rot_from_q = q.toRotationMatrix()

    assert is_same_matrix(m_rot, m_rot_from_q), "Transformation matrices are not equal"


def test_angle_axis_directly():
    angle_restored, axis_restored = M_orig.to_angle_axis()
    assert np.allclose(axis_orig.data, axis_restored.data), f"Original axis {axis_orig}, restored {axis_restored}"
    assert np.allclose(angle_orig, angle_restored), f"Original angle {angle_orig}, restored {angle_restored}"


def test_angle_axis():
    q_restored = rotation_matrix_to_quaternion(M_orig)
    assert np.allclose(q_restored.q, q_orig.q), f"Original quaternion {q_orig}, restored {q_restored}"

    angle_restored, axis_restored = q_restored.to_angle_axis()
    assert np.allclose(axis_orig.data, axis_restored.data), f"Original axis {axis_orig}, restored {angle_restored}"
    assert np.allclose(angle_orig, angle_restored), f"Original angle {angle_orig}, restored {angle_restored}"

