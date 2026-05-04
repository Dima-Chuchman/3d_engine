import pytest
import numpy as np
from src.math.Mat4x4 import Mat4x4
from src.math.Vec3 import Vec3
from src.math.Vec4 import Vec4

# Matrix for testing
IDENTITY = Mat4x4.identity()

# 1. Множення Mat4x4 @ Mat4x4
def test_matmul_with_mat4x4():
    m1 = Mat4x4.identity()
    m2 = Mat4x4.identity()
    result = m1 @ m2
    assert isinstance(result, Mat4x4)
    assert np.allclose(result.data, np.eye(4))

# 2. Множення Mat4x4 @ Vec4
def test_matmul_with_vec4():
    v = Vec4(1, 2, 3, 1)
    result = IDENTITY @ v
    assert isinstance(result, Vec4)
    assert np.allclose(result.data, v.data)

# 3. Множення Mat4x4 @ Vec3
def test_matmul_with_vec3():
    v = Vec3(1, 2, 3)
    result = IDENTITY @ v
    expected = Vec4(1, 2, 3, 0)
    assert isinstance(result, Vec4)
    assert np.allclose(result.data, expected.data)

# 4. Множення Mat4x4 @ np.ndarray (4x4)
def test_matmul_with_ndarray_4x4():
    arr = np.eye(4)
    result = IDENTITY @ arr
    assert isinstance(result, Mat4x4)
    assert np.allclose(result.data, np.eye(4))

# 5. Множення Mat4x4 @ np.ndarray (4,)
def test_matmul_with_ndarray_4vec():
    arr = np.array([1, 2, 3, 1], dtype=float)
    result = IDENTITY @ arr
    assert isinstance(result, Vec4)
    assert np.allclose(result.data, arr)

# 6. Множення Mat4x4 @ np.ndarray неправильної форми → TypeError
def test_matmul_with_ndarray_wrong_shape():
    arr = np.array([1, 2, 3])  # only 3 elements
    with pytest.raises(TypeError):
        IDENTITY @ arr

# 7. Множення Mat4x4 @ np.ndarray форми (3, 3) → TypeError
def test_matmul_with_3x3_array():
    arr = np.eye(3)
    with pytest.raises(TypeError):
        IDENTITY @ arr

# 8. Множення Mat4x4 @ unsupported type (str)
def test_matmul_with_invalid_type_string():
    with pytest.raises(TypeError):
        IDENTITY @ "hello"

# 9. Множення Mat4x4 @ unsupported type (list)
def test_matmul_with_invalid_type_list():
    with pytest.raises(TypeError):
        IDENTITY @ [1, 2, 3, 4]

# 10. Перевірка, що вираз використовує саме __matmul__, а не __mul__
def test_matmul_operator_resolution():
    m1 = Mat4x4.identity()
    m2 = Mat4x4.identity()
    result = m1 @ m2
    assert hasattr(result, "data")
