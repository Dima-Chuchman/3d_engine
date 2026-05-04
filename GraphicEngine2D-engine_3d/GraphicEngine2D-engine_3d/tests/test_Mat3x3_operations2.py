import pytest
import numpy as np
from src.math.Mat3x3 import Mat3x3
from src.math.Vec3 import Vec3

IDENTITY = Mat3x3()  # identity 3x3 matrix

# 1. Mat3x3 @ Mat3x3
def test_matmul_with_mat3x3():
    m1 = Mat3x3()
    m2 = Mat3x3()
    result = m1 @ m2
    assert isinstance(result, Mat3x3)
    assert np.allclose(result.data, np.eye(3))

# 2. Mat3x3 @ Vec3
def test_matmul_with_vec3():
    v = Vec3(1, 2, 3)
    result = IDENTITY @ v
    assert isinstance(result, Vec3)
    assert np.allclose(result.data, v.data)

# 3. Mat3x3 @ ndarray (3, 3)
def test_matmul_with_ndarray_3x3():
    arr = np.eye(3)
    result = IDENTITY @ arr
    assert isinstance(result, Mat3x3)
    assert np.allclose(result.data, arr)

# 4. Mat3x3 @ ndarray (3,)
def test_matmul_with_ndarray_3vec():
    arr = np.array([4, 5, 6])
    result = IDENTITY @ arr
    assert isinstance(result, Vec3)
    assert np.allclose(result.data, arr)

# 5. Mat3x3 @ ndarray неправильної форми (2,)
def test_matmul_with_ndarray_wrong_shape_vector():
    arr = np.array([1, 2])
    with pytest.raises(TypeError):
        IDENTITY @ arr

# 6. Mat3x3 @ ndarray неправильної форми (2x2)
def test_matmul_with_ndarray_wrong_shape_matrix():
    arr = np.eye(2)
    with pytest.raises(TypeError):
        IDENTITY @ arr

# 7. Mat3x3 @ рядок (str)
def test_matmul_with_string():
    with pytest.raises(TypeError):
        IDENTITY @ "invalid"

# 8. Mat3x3 @ список (list)
def test_matmul_with_list():
    with pytest.raises(TypeError):
        IDENTITY @ [1, 2, 3]

# 9. Mat3x3 @ множина (set)
def test_matmul_with_set():
    with pytest.raises(TypeError):
        IDENTITY @ {1, 2, 3}

# 10. Mat3x3 @ None
def test_matmul_with_none():
    with pytest.raises(TypeError):
        IDENTITY @ None
