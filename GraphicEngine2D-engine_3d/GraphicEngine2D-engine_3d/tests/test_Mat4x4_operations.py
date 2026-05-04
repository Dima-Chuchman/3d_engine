import pytest
import numpy as np
from src.math.Mat4x4 import Mat4x4
from src.math.Vec4 import Vec4
from src.math.Vec3 import Vec3

# ========================  ARITHMETIC OPERATIONS  ===========================
# 1. Test matrix addition
def test_matrix_addition():
    mat1 = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    mat2 = Mat4x4(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    result = mat1 + mat2
    expected = np.full((4, 4), 17, dtype=float)  # Sum of corresponding elements
    assert np.allclose(result.data, expected), "Error in matrix addition"

# 2. Test matrix subtraction
def test_matrix_subtraction():
    mat1 = Mat4x4(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
    mat2 = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    result = mat1 - mat2
    expected = np.full((4, 4), 10, dtype=float) - np.arange(1, 17).reshape(4, 4)
    assert np.allclose(result.data, expected), "Error in matrix subtraction"

# 3. Test unary minus
def test_matrix_negation():
    mat = Mat4x4(1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16)
    result = -mat
    expected = -mat.data
    assert np.allclose(result.data, expected), "Error in matrix sign change"

# ========================  MULTIPLICATION  ===========================
# 4. Test matrix-matrix multiplication
def test_matrix_multiplication():
    mat1 = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    mat2 = Mat4x4.identity()
    result = mat1 @ mat2
    assert np.allclose(result.data, mat1.data), "Error in multiplying matrix by identity"

# 5. Test matrix-Vec4 multiplication
def test_matrix_vector_multiplication():
    mat = Mat4x4.identity()
    vec = Vec4(3, 4, 5, 1)
    result = mat @ vec
    assert np.allclose(result.data, vec.data), "Error in matrix-vector multiplication"

# 6. Test matrix-Vec3 multiplication
def test_matrix_vector3_multiplication():
    mat = Mat4x4.identity()
    vec = Vec3(3, 4, 5)
    result = mat * vec  # Should automatically convert Vec3 to Vec4 (with w=0)
    expected = Vec4(3, 4, 5)
    assert np.allclose(result.data, expected.data), "Error in matrix-Vec3 multiplication"

# ========================  INVERSION AND TRANSPOSITION  ===========================
# 7. Test finding inverse matrix
def test_matrix_inverse():
    mat = Mat4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    inverse_mat = mat.inverse()
    expected = np.linalg.inv(mat.data)
    assert np.allclose(inverse_mat.data, expected), "Error in finding inverse matrix"

# 8. Test matrix transposition
def test_matrix_transpose():
    mat = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    transposed = mat.transpose()
    expected = mat.data.T
    assert np.allclose(transposed.data, expected), "Error in matrix transposition"

# ========================  ROTATION OPERATIONS  ===========================
# 9. Test rotation around X
def test_rotation_x():
    mat = Mat4x4.rotation_x(90, is_radians=False)
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected, atol=1e-6), "Error in rotation matrix around X"

# 10. Test rotation around Y
def test_rotation_y():
    mat = Mat4x4.rotation_y(90, is_radians=False)
    expected = np.array([[0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [-1, 0, 0, 0],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected, atol=1e-6), "Error in rotation matrix around Y"

# 11. Test rotation around Z
def test_rotation_z():
    mat = Mat4x4.rotation_z(90, is_radians=False)
    expected = np.array([[0, -1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected, atol=1e-6), "Error in rotation matrix around Z"

# ========================  SCALING AND TRANSLATION  ===========================
# 12. Test scaling
def test_matrix_scaling():
    mat = Mat4x4.scale(2, 3, 4)
    expected = np.array([[2, 0, 0, 0],
                         [0, 3, 0, 0],
                         [0, 0, 4, 0],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected), "Error in scale matrix"

# 13. Test translation
def test_matrix_translation():
    mat = Mat4x4.translation(3, 4, 5)
    expected = np.array([[1, 0, 0, 3],
                         [0, 1, 0, 4],
                         [0, 0, 1, 5],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected), "Error in translation matrix"

# ========================  CONTINUED TESTS  ===========================

# 14. Test множення матриці на скаляр (недозволена операція)
def test_matrix_scalar_multiplication():
    mat = Mat4x4.identity()
    with pytest.raises(TypeError, match="Multiplication is only possible with other Matrix4x4 objects or numpy.ndarray 4x4"):
        m2 = mat * 2  # Error: multiplication by scalar is not supported

# 15. Test знаходження норми матриці
def test_matrix_norm():
    mat = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    expected = np.linalg.norm(mat.data)
    assert np.isclose(mat.norm(), expected), "Error in finding matrix norm"

# 16. Test помилки при знаходженні оберненої матриці, якщо вона вироджена
def test_singular_matrix_inverse():
    singular_mat = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)  # Determinant = 0
    with pytest.raises(ValueError, match="Matrix has no inverse"):
        singular_mat.inverse()

# 17. Test правильності транспонування через `.T`
def test_matrix_transpose_property():
    mat = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    transposed = mat.T  # Using the property
    expected = mat.data.T
    assert np.allclose(transposed.data, expected), "Error in transposition via .T"

# 18. Test обертання у `Euler`-координатах (XYZ)
def test_rotation_euler_xyz():
    mat = Mat4x4.rotation_euler(30, 45, 60, configuration="xyz")
    assert mat is not None, "Error: Euler rotation does not work"

# 19. Test обертання у `Euler`-координатах (ZXZ)
def test_rotation_euler_zxz():
    mat = Mat4x4.rotation_euler(90, 45, 30, configuration="zxz")
    assert mat is not None, "Error: Euler rotation (ZXZ) does not work"

# 20. Test помилки для невідомої конфігурації `Euler`-обертання
def test_invalid_rotation_euler():
    with pytest.raises(ValueError, match=Mat4x4.ERROR_MESSAGE_EULER_CONFIG_UNKNOWN):
        Mat4x4.rotation_euler(30, 45, 60, configuration="abc")

# 21. Test `toEuler()` для конфігурації XYZ
def test_toEuler_xyz():
    mat = Mat4x4.rotation_euler(30, 45, 60, configuration="xyz")
    angles = mat.toEuler("XYZ")
    assert len(angles) == 3, "Error getting Euler coordinates"

# 22. Test `toEuler()` для конфігурації ZXZ
def test_toEuler_zxz():
    mat = Mat4x4.rotation_euler(30, 45, 60, configuration="zxz")
    angles = mat.toEuler("ZXZ")
    assert len(angles) == 3, "Error getting Euler coordinates (ZXZ)"

# 23. Test помилки `toEuler()` для невідомої конфігурації
def test_toEuler_invalid():
    mat = Mat4x4.identity()
    with pytest.raises(ValueError, match=Mat4x4.ERROR_MESSAGE_EULER_CONFIG_UNKNOWN):
        mat.toEuler("ABC")

# 24. Test масштабування з одним числом (рівномірний скейлінг)
def test_matrix_uniform_scaling():
    mat = Mat4x4.scale(3)
    expected = np.array([[3, 0, 0, 0],
                         [0, 3, 0, 0],
                         [0, 0, 3, 0],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected), "Error in uniform scaling"

# 25. Test масштабування з `Vec3`
def test_matrix_scaling_vec3():
    vec = Vec3(2, 3, 4)
    mat = Mat4x4.scale(vec)
    expected = np.array([[2, 0, 0, 0],
                         [0, 3, 0, 0],
                         [0, 0, 4, 0],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected), "Error in Vec3 scaling"

# 26. Test помилки для некоректного масштабування
def test_invalid_matrix_scaling():
    with pytest.raises(ValueError, match=Mat4x4.ERROR_MESSAGE_SCALE):
        Mat4x4.scale("invalid")

# 27. Test трансляції за допомогою `Vec3`
def test_matrix_translation_vec3():
    vec = Vec3(3, 4, 5)
    mat = Mat4x4.translation(vec)
    expected = np.array([[1, 0, 0, 3],
                         [0, 1, 0, 4],
                         [0, 0, 1, 5],
                         [0, 0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected), "Error in Vec3 translation"

# 28. Test множення матриці на `numpy.ndarray`
def test_matrix_numpy_multiplication():
    mat = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    array = np.eye(4, dtype=float)
    result = mat @ array
    expected = np.dot(mat.data, array)
    assert np.allclose(result.data, expected), "Error in matrix multiplication with numpy array"

# 29. Test помилки при множенні на `numpy.ndarray` неправильного розміру
def test_invalid_numpy_multiplication():
    mat = Mat4x4.identity()
    invalid_array = np.eye(3)  # Invalid size (3×3 instead of 4×4)
    with pytest.raises(TypeError, match=Mat4x4.ERROR_MESSAGE_MULT):
        mat @ invalid_array

# 30. Test помилки при обертанні навколо некоректного вектора
def test_invalid_rotation_vector():
    with pytest.raises(ValueError, match=Mat4x4.ERROR_MESSAGE_ROTATION):
        Mat4x4.rotation(45, [1, 2])  # Invalid vector size
