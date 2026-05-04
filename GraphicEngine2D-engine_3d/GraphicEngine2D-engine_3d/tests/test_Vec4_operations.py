import pytest
import numpy as np
from src.math.Vec3 import Vec3
from src.math.Vec4 import Vec4, vertex

# 1. Test множення вектора на нуль (результат має бути нульовим вектором)
def test_vector_multiplication_by_zero():
    vec = Vec4(1, -2, 3, -4)
    result = vec * 0
    expected = np.zeros(4, dtype=float)
    assert np.allclose(result.data, expected), "Error: multiplication by 0 should give (0,0,0,0)"

# 2. Test множення нульового вектора на скаляр (має залишатися нульовим)
def test_zero_vector_multiplication():
    vec = Vec4()
    result = vec * 5
    expected = np.zeros(4, dtype=float)
    assert np.allclose(result.data, expected), "Error: multiplying zero vector by number should give (0,0,0,0)"

# 3. Test порівняння двох однакових векторів
def test_vector_equality():
    vec1 = Vec4(1, 2, 3, 4)
    vec2 = Vec4(1, 2, 3, 4)
    assert np.allclose(vec1.data, vec2.data), "Error: vectors should be equal"

# 4. Test порівняння двох різних векторів
def test_vector_inequality():
    vec1 = Vec4(1, 2, 3, 4)
    vec2 = Vec4(4, 3, 2, 1)
    assert not np.allclose(vec1.data, vec2.data), "Error: vectors should not be equal"

# 5. Test множення одиничного вектора на себе (скалярний добуток має бути 1)
def test_unit_vector_dot_product():
    vec = Vec4(1, 0, 0, 0)
    result = vec * vec
    expected = 1
    assert np.isclose(result, expected), "Error: dot product should be 1"

# 6. Test скалярного добутку ортогональних векторів (має бути 0)
def test_orthogonal_dot_product():
    vec1 = Vec4(1, 0, 0, 0)
    vec2 = Vec4(0, 1, 0, 0)
    result = vec1 * vec2
    expected = 0
    assert np.isclose(result, expected), "Error: dot product of orthogonal vectors should be 0"

# 7. Test помилки при некоректному додаванні (наприклад, рядок)
def test_invalid_vector_addition():
    vec = Vec4(1, 2, 3, 4)
    with pytest.raises(TypeError):
        result = vec + "invalid"

# 8. Test помилки при некоректному множенні (наприклад, рядок)
def test_invalid_vector_multiplication():
    vec = Vec4(1, 2, 3, 4)
    with pytest.raises(TypeError):
        result = vec * "invalid"

# 9. Test зміни координати x
def test_vector_set_x():
    vec = Vec4(0, 0, 0, 0)
    vec.x = 10
    assert vec.x == 10, "Error: x should be 10"

# 10. Test зміни координати w
def test_vector_set_w():
    vec = Vec4(1, 2, 3, 4)
    vec.w = 100
    assert vec.w == 100, "Error: w should be 100"

# 11. Test vector normalization (збереження напрямку)
def test_vector_normalization_preserves_direction():
    vec = Vec4(3, 4, 0, 0)
    norm_vec = vec.normalized()
    expected_direction = np.array([3, 4, 0, 0]) / np.linalg.norm([3, 4, 0, 0])
    assert np.allclose(norm_vec.data, expected_direction), "Error: normalization should preserve direction"

# 12. Test нормування нульового вектора (повертає нульовий вектор)
def test_zero_vector_normalization():
    vec = Vec4()
    norm_vec = vec.normalized()
    expected = np.zeros(4, dtype=float)
    assert np.allclose(norm_vec.data, expected), "Error: normalizing zero vector should give (0,0,0,0)"

# 13. Test копіювання вектора через конструктор
def test_vector_copy():
    vec1 = Vec4(5, -3, 2, 1)
    vec2 = Vec4(vec1)
    assert np.allclose(vec1.data, vec2.data), "Error: copy must be identical to original"

# 14. Test створення `Vec4` з `Vec3` (має додати `0.0`)
def test_vec4_from_vec3():
    v3 = Vec3(3, 4, 5)
    v4 = Vec4(v3)
    expected = np.array([3, 4, 5, 0], dtype=float)
    assert np.allclose(v4.data, expected), "Error: creating Vec4 from Vec3 should append 0.0"

# 15. Test від’ємного значення `Vec4`
def test_vector_negate():
    vec = Vec4(1, -2, 3, -4)
    result = -vec
    expected = np.array([-1, 2, -3, 4], dtype=float)
    assert np.allclose(result.data, expected), "Error: -Vec4 should be (-1, 2, -3, 4)"

# 16. Test для `xyzw` (має повернути всі координати)
def test_xyzw_property():
    vec = Vec4(7, 8, 9, 10)
    expected = np.array([7, 8, 9, 10], dtype=float)
    assert np.allclose(vec.xyzw, expected), "Error: xyzw should return (7, 8, 9, 10)"

# 17. Test для `xyz` (має повернути перші три координати)
def test_xyz_property():
    vec = Vec4(7, 8, 9, 10)
    expected = np.array([7, 8, 9], dtype=float)
    assert np.allclose(vec.xyz, expected), "Error: xyz should return (7, 8, 9)"

# 18. Test для `xy`
def test_xy_property():
    vec = Vec4(7, 8, 9, 10)
    expected = np.array([7, 8], dtype=float)
    assert np.allclose(vec.xy, expected), "Error: xy should return (7, 8)"

# 19. Test для `xz`
def test_xz_property():
    vec = Vec4(7, 8, 9, 10)
    expected = np.array([7, 9], dtype=float)
    assert np.allclose(vec.xz, expected), "Error: xz should return (7, 9)"

# 20. Test для `yz`
def test_yz_property():
    vec = Vec4(7, 8, 9, 10)
    expected = np.array([8, 9], dtype=float)
    assert np.allclose(vec.yz, expected), "Error: yz should return (8, 9)"
