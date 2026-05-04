import pytest
import numpy as np
from src.math.Vec3 import Vec3, vertex  # Make sure the import path is correct!

# 1. Створення вектора з двох чисел (третьою координатою має бути 0)
def test_vector_from_two_numbers():
    vec = Vec3(1, 2)
    expected = np.array([1, 2, 0], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (1, 2, 0)"

# 2. Створення точки через статичний метод vertex()
def test_vertex():
    vec = vertex(3, 4)
    expected = np.array([3, 4, 1], dtype=float)
    assert np.allclose(vec.data, expected), "Error: vertex() should create vector (3, 4, 1)"

# 3. Створення точки vertex() без аргументів (має бути (0, 0, 1))
def test_vertex_default():
    vec = vertex()
    expected = np.array([0, 0, 1], dtype=float)
    assert np.allclose(vec.data, expected), "Error: vertex() without arguments should create (0, 0, 1)"

# 4. Перевірка нормалізації вектора (довжина має стати 1)
def test_vector_normalization():
    vec = Vec3(3, 4, 0)
    vec.normalize()
    expected_length = 1.0
    assert np.isclose(vec.norm(), expected_length), "Error: normalized vector should have length 1"

# 5. Перевірка повернення нового нормалізованого вектора (метод normalized)
def test_vector_normalized():
    vec = Vec3(3, 4, 0)
    normalized_vec = vec.normalized()
    expected_length = 1.0
    assert np.isclose(normalized_vec.norm(), expected_length), "Error: normalized vector should have length 1"
    assert not np.allclose(vec.data, normalized_vec.data), "Original vector must not change"

# 6. Перевірка векторного добутку (cross product)
def test_vector_cross_product():
    vec1 = Vec3(1, 0, 0)
    vec2 = Vec3(0, 1, 0)
    cross_product = vec1.cross(vec2)
    expected = np.array([0, 0, 1], dtype=float)
    assert np.allclose(cross_product.data, expected), "Error: cross product should be (0, 0, 1)"

# 7. Перевірка помилки при некоректному вхідному значенні в cross()
def test_vector_cross_invalid():
    vec = Vec3(1, 0, 0)
    with pytest.raises(TypeError, match=Vec3.ERROR_MESSAGE_CROSS):
        vec.cross([1, 2])  # Invalid vector size

# 8. Перевірка доступу до x, y, z через властивості
def test_vector_properties():
    vec = Vec3(5, 6, 7)
    assert vec.x == 5, "Error: x має бути 5"
    assert vec.y == 6, "Error: y має бути 6"
    assert vec.z == 7, "Error: z має бути 7"

# 9. Перевірка встановлення значень через властивості x, y, z
def test_vector_set_properties():
    vec = Vec3(0, 0, 0)
    vec.x = 9
    vec.y = 8
    vec.z = 7
    expected = np.array([9, 8, 7], dtype=float)
    assert np.allclose(vec.data, expected), "Error: x, y, z values should update the vector"

# 10. Перевірка правильного повернення підвекторів xy, xz, yz
def test_vector_subvectors():
    vec = Vec3(1, 2, 3)
    assert np.allclose(vec.xy, [1, 2]), "Error: xy should be (1, 2)"
    assert np.allclose(vec.xz, [1, 3]), "Error: xz should be (1, 3)"
    assert np.allclose(vec.yz, [2, 3]), "Error: yz should be (2, 3)"
