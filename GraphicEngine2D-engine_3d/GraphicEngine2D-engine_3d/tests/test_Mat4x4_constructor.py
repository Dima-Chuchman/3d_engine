import numpy as np

from src.math.Mat3x3 import Mat3x3
from src.math.Mat4x4 import Mat4x4
from src.math.Vec4 import Vec4


# 1. Test створення одиничної матриці (без аргументів)
def test_default_matrix():
    mat = Mat4x4()
    expected = np.eye(4, dtype=float)
    assert np.allclose(mat.data, expected), "Error: should be identity matrix"

# 2. Test створення з 16 чисел
def test_matrix_from_16_elements():
    mat = Mat4x4(1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 10, 11, 12,
                 13, 14, 15, 16
                 )
    expected = np.array(range(1, 17), dtype=float).reshape(4, 4)
    assert np.allclose(mat.data, expected), "Error: should be 4×4 matrix"

# 3. Test створення з 9 чисел (має доповнювати до 4×4)
def test_matrix_from_9_elements():
    mat = Mat4x4(1, 2, 3, 4, 5, 6, 7, 8, 9)
    expected = np.eye(4, dtype=float)
    expected[:3, :3] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    assert np.allclose(mat.data, expected), "Error: should be 3×3 padded to 4×4"

# 4. Test створення з `Mat3x3` (має доповнити до 4×4)
def test_matrix_from_Mat3x3():
    mat3x3 = Mat3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    mat4x4 = Mat4x4(mat3x3)
    expected = np.eye(4, dtype=float)
    expected[:3, :3] = mat3x3.data
    assert np.allclose(mat4x4.data, expected), "Error: should be padded 4×4 matrix"

# 5. Test створення з `Mat4x4` (має бути копія)
def test_matrix_from_Mat4x4():
    original = Mat4x4(1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 10, 11, 12,
                 13, 14, 15, 16
                 )
    copy = Mat4x4(original)
    assert np.allclose(copy.data, original.data), "Copy must be identical to original"

# 6. Test створення з 2×2 матриці (має доповнити до 4×4)
def test_matrix_from_4_elements():
    mat = Mat4x4(1, 2, 3, 4)
    expected = np.eye(4, dtype=float)
    expected[:2, :2] = np.array([[1, 2], [3, 4]], dtype=float)
    assert np.allclose(mat.data, expected), "Error: should be 2×2 padded to 4×4"

# 7. Test створення з 4 векторів `Vec4`
def test_matrix_from_four_vec4():
    v1, v2, v3, v4 = Vec4(1, 2, 3, 4), Vec4(5, 6, 7, 8), Vec4(9, 10, 11, 12), Vec4(13, 14, 15, 16)
    mat = Mat4x4(v1, v2, v3, v4)
    expected = np.vstack([v1.data, v2.data, v3.data, v4.data])
    assert np.allclose(mat.data, expected), "Error: should be 4×4 matrix з 4 Vec4"

# 8. Test створення з 4 масивів `numpy`
def test_matrix_from_four_numpy_arrays():
    v1, v2, v3, v4 = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12]), np.array([13, 14, 15, 16])
    mat = Mat4x4(v1, v2, v3, v4)
    expected = np.vstack([v1, v2, v3, v4])
    assert np.allclose(mat.data, expected), "Error: should be 4×4 matrix з 4 numpy векторів"

# 9. Test створення з `numpy.ndarray` 4×4
def test_matrix_from_numpy_array():
    data = np.arange(16).reshape(4, 4)
    mat = Mat4x4(data)
    assert np.allclose(mat.data, data), "Error: should be 4×4 matrix з numpy array"

# 10. Test створення з `numpy.ndarray` 3×3 (автоматичне доповнення)
def test_matrix_from_numpy_3x3():
    data = np.arange(9).reshape(3, 3)
    mat = Mat4x4(data)
    expected = np.eye(4, dtype=float)
    expected[:3, :3] = data
    assert np.allclose(mat.data, expected), "Error: should be 3×3 padded to 4×4"
