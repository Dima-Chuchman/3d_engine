import pytest
import numpy as np
from src.math.Vec3 import Vec3
from src.math.Vec4 import Vec4, vertex

# 1. Test creating zero vector
def test_default_vector():
    vec = Vec4()
    expected = np.zeros(4, dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be zero vector"

# 2. Test creating from 3 numbers (last coordinate should be 0)
def test_vector_from_three_numbers():
    vec = Vec4(1, 2, 3)
    expected = np.array([1, 2, 3, 0], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (1, 2, 3, 0)"

# 3. Test creating from 4 numbers
def test_vector_from_four_numbers():
    vec = Vec4(1, 2, 3, 4)
    expected = np.array([1, 2, 3, 4], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (1, 2, 3, 4)"

# 4. Test creating from Vec3 (should automatically append 0.0)
def test_vector_from_vec3():
    v3 = Vec3(1, 2, 3)
    vec = Vec4(v3)
    expected = np.array([1, 2, 3, 0], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (1, 2, 3, 0) created via Vec3"

# 5. Test creating from Vec4 (should be a copy)
def test_vector_from_vec4():
    original = Vec4(4, 5, 6, 7)
    copy = Vec4(original)
    assert np.allclose(copy.data, original.data), "Vector copy must be identical to original"

# 6. Test creating via vertex()
def test_vector_vertex():
    vec = vertex(3, 4, 5)
    expected = np.array([3, 4, 5, 1], dtype=float)
    assert np.allclose(vec.data, expected), "Error: vertex() should create vector (3, 4, 5, 1)"

# 7. Test adding two Vec4
def test_vector_addition():
    vec1 = Vec4(1, 2, 3, 4)
    vec2 = Vec4(4, 3, 2, 1)
    result = vec1 + vec2
    expected = np.array([5, 5, 5, 5], dtype=float)
    assert np.allclose(result.data, expected), "Error: addition should give (5, 5, 5, 5)"

# 8. Test subtracting two Vec4
def test_vector_subtraction():
    vec1 = Vec4(1, 2, 3, 4)
    vec2 = Vec4(4, 3, 2, 1)
    result = vec1 - vec2
    expected = np.array([-3, -1, 1, 3], dtype=float)
    assert np.allclose(result.data, expected), "Error: subtraction should give (-3, -1, 1, 3)"

# 9. Test scalar multiplication
def test_vector_scalar_multiplication():
    vec = Vec4(1, 2, 3, 4)
    result = vec * 2
    expected = np.array([2, 4, 6, 8], dtype=float)
    assert np.allclose(result.data, expected), "Error: multiplication by 2 should give (2, 4, 6, 8)"

# 10. Test dot product of two vectors
def test_vector_dot_product():
    vec1 = Vec4(1, 2, 3, 4)
    vec2 = Vec4(4, 3, 2, 1)
    result = vec1 * vec2
    expected = 1*4 + 2*3 + 3*2 + 4*1  # = 20
    assert np.isclose(result, expected), f"Error: expected dot product {expected}, got {result}"

# 11. Test vector normalization
def test_vector_normalization():
    vec = Vec4(3, 4, 0, 0)
    vec.normalize()
    expected_length = 1.0
    assert np.isclose(vec.norm(), expected_length), "Error: normalized vector should have length 1"

# 12. Test returning new normalized vector (method normalized)
def test_vector_normalized():
    vec = Vec4(3, 4, 0, 0)
    normalized_vec = vec.normalized()
    expected_length = 1.0
    assert np.isclose(normalized_vec.norm(), expected_length), "Error: normalized vector should have length 1"
    assert not np.allclose(vec.data, normalized_vec.data), "Original vector must not change"

# 13. Test від’ємного вектора
def test_vector_negation():
    vec = Vec4(1, -2, 3, -4)
    result = -vec
    expected = np.array([-1, 2, -3, 4], dtype=float)
    assert np.allclose(result.data, expected), "Error: від’ємний вектор має бути (-1, 2, -3, 4)"

# 14. Test getting and setting x, y, z, w
def test_vector_properties():
    vec = Vec4(1, 2, 3, 4)
    vec.x = 10
    vec.y = 20
    vec.z = 30
    vec.w = 40
    expected = np.array([10, 20, 30, 40], dtype=float)
    assert np.allclose(vec.data, expected), "Error: set values x, y, z, w should be (10, 20, 30, 40)"

# 15. Test access to sub-vectors xy, xz, yz, xyz, xyzw
def test_vector_subvectors():
    vec = Vec4(1, 2, 3, 4)
    assert np.allclose(vec.xy, [1, 2]), "Error: xy should be (1, 2)"
    assert np.allclose(vec.xz, [1, 3]), "Error: xz should be (1, 3)"
    assert np.allclose(vec.yz, [2, 3]), "Error: yz should be (2, 3)"
    assert np.allclose(vec.xyz, [1, 2, 3]), "Error: xyz should be (1, 2, 3)"
    assert np.allclose(vec.xyzw, [1, 2, 3, 4]), "Error: xyzw should be (1, 2, 3, 4)"
