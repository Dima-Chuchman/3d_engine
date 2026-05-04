import pytest
import numpy as np
from src.math.Mat3x3 import Mat3x3
from src.math.Vec3 import Vec3
from src.math.Rotations import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
from src.math.Scale import scale_matrix
from src.math.Translation import translation_matrix2d


# 1. Test matrix addition
def test_matrix_addition():
    mat1 = Mat3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    mat2 = Mat3x3(9, 8, 7, 6, 5, 4, 3, 2, 1)
    result = mat1 + mat2
    expected = np.array(mat1.data + mat2.data)

    assert np.allclose(result.data, expected), f"Error in addition: expected {expected}, got {result.data}"


# 2. Test matrix-matrix multiplication
def test_matrix_multiplication():
    mat1 = Mat3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    mat2 = Mat3x3(9, 8, 7, 6, 5, 4, 3, 2, 1)
    result = mat1 @ mat2
    expected = np.dot(mat1.data, mat2.data)

    assert np.allclose(result.data, expected), f"Error in matrix multiplication: expected {expected}, got {result.data}"


# 3. Test matrix-vector multiplication
def test_matrix_vector_multiplication():
    mat = Mat3x3.identity()
    vec = Vec3(3, 4, 5)
    result = mat @ vec
    expected = vec.data  # Since this is identity matrix, vector does not change

    assert np.allclose(result.data,
                       expected), f"Error: multiplication result should be {expected}, got {result.data}"


# 4. Test finding inverse matrix
def test_matrix_inverse():
    mat = Mat3x3(2, -1, 0, -1, 2, -1, 0, -1, 2)
    inverse_mat = mat.inverse()
    expected = np.linalg.inv(mat.data)

    assert np.allclose(inverse_mat.data,
                       expected), f"Error: inverse matrix should be {expected}, got {inverse_mat.data}"


# 5. Test error when finding inverse of singular matrix
def test_singular_matrix_inverse():
    singular_mat = Mat3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)  # Determinant = 0
    # singular_mat = Mat3x3(1, 1, 1, 4, 4, 4, 7, 8, 9)  # Determinant = 0

    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_INV_DOESNT_EXIST):
        singular_mat.inverse()


# 6. Test identity matrix
def test_matrix_identity():
    identity_mat = Mat3x3.identity()
    expected = np.eye(3, dtype=float)

    assert np.allclose(identity_mat.data, expected), "Error: identity matrix should be standard"

# 6.1. Test identity matrix
def test_matrix_identity2():
    identity_mat = Mat3x3()
    expected = np.eye(3, dtype=float)

    assert np.allclose(identity_mat.data, expected), "Error: identity matrix should be standard"


# 7. Test creating rotation matrix around X
def test_rotation_x():
    mat = Mat3x3.rotation_x(90, is_radians=False)
    expected = rotation_matrix_x(np.radians(90))

    assert np.allclose(mat.data,
                       expected), f"Error: rotation matrix around X should be {expected}, got {mat.data}"


# 8. Test creating rotation matrix around Y
def test_rotation_y():
    mat = Mat3x3.rotation_y(90, is_radians=False)
    expected = rotation_matrix_y(np.radians(90))

    assert np.allclose(mat.data,
                       expected), f"Error: rotation matrix around Y should be {expected}, got {mat.data}"


# 9. Test creating rotation matrix around Z
def test_rotation_z():
    mat = Mat3x3.rotation_z(90, is_radians=False)
    expected = rotation_matrix_z(np.radians(90))

    assert np.allclose(mat.data,
                       expected), f"Error: rotation matrix around Z should be {expected}, got {mat.data}"


# 10. Test creating translation matrix
def test_matrix_translation():
    mat = Mat3x3.translation(3, 4)
    expected = translation_matrix2d(3, 4)

    assert np.allclose(mat.data, expected), f"Error: translation matrix should be {expected}, got {mat.data}"


# 11. Test creating scale matrix with two parameters
def test_matrix_scaling_two_values():
    mat = Mat3x3.scale(2, 3)
    expected = scale_matrix(2, 3)

    assert np.allclose(mat.data, expected), f"Error: scale matrix should be {expected}, got {mat.data}"


# 12. Test creating scale matrix with one parameter (uniform scale)
def test_matrix_scaling_one_value():
    mat = Mat3x3.scale(2)
    expected = scale_matrix(2, 2, 2)

    assert np.allclose(mat.data, expected), f"Error: scale matrix should be {expected}, got {mat.data}"


# 13. Test creating scale matrix with vector
def test_matrix_scaling_with_vector():
    vec = Vec3(2, 3, 1)
    mat = Mat3x3.scale(vec)
    expected = scale_matrix(2, 3, 1)

    assert np.allclose(mat.data, expected), f"Error: scale matrix should be {expected}, got {mat.data}"


# 14. Test error for invalid scaling
def test_invalid_matrix_scaling():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_SCALE):
        Mat3x3.scale("invalid")


# 15. Test multiplication by numpy array (allowed)
def test_matrix_numpy_multiplication():
    mat = Mat3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = mat @ array
    expected = np.dot(mat.data, array)

    assert np.allclose(result.data,
                       expected), f"Error: multiplication result should be {expected}, got {result.data}"
