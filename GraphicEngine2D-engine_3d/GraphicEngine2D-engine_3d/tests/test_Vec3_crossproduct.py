# 1. Векторний добуток нетривіальних векторів
import numpy as np

from src.math.Vec3 import Vec3


# 1. Векторний добуток нетривіальних векторів
def test_vector_cross_product_nontrivial():
    vec1 = Vec3(2, 3, 4)
    vec2 = Vec3(5, 6, 7)
    cross_product = vec1.cross(vec2)
    expected = np.array([-3, 6, -3], dtype=float)  # Manual computation: (3*7 - 4*6, 4*5 - 2*7, 2*6 - 3*5)

    assert np.allclose(cross_product.data, expected), f"Expected (-3, 6, -3), got {cross_product.data}"

# 2. Векторний добуток унікальних векторів (не ортогональних, не паралельних)
def test_vector_cross_product_unique():
    vec1 = Vec3(-1, 2, -3)
    vec2 = Vec3(4, -5, 6)
    cross_product = vec1.cross(vec2)
    expected = np.array([-3, -6, -3], dtype=float)  # Correct value

    assert np.allclose(cross_product.data, expected), f"Expected (-3, -6, -3), got {cross_product.data}"

# 3. Векторний добуток паралельних векторів (має бути [0,0,0])
def test_vector_cross_product_parallel():
    vec1 = Vec3(1, 2, 3)
    vec2 = Vec3(2, 4, 6)  # Vector proportional to the first one
    cross_product = vec1.cross(vec2)
    expected = np.array([0, 0, 0], dtype=float)

    assert np.allclose(cross_product.data, expected), f"Expected (0,0,0), got {cross_product.data}"


# 4. Векторний добуток антипаралельних векторів (також має бути [0,0,0])
def test_vector_cross_product_antiparallel():
    vec1 = Vec3(3, -3, 1)
    vec2 = Vec3(-6, 6, -2)  # -2 * vec1 (anti-parallel)
    cross_product = vec1.cross(vec2)
    expected = np.array([0, 0, 0], dtype=float)

    assert np.allclose(cross_product.data, expected), f"Expected (0,0,0), got {cross_product.data}"


# 5. Векторний добуток нетривіальних векторів (переставлення аргументів має змінити знак)
def test_vector_cross_product_order():
    vec1 = Vec3(7, -8, 9)
    vec2 = Vec3(-1, 2, -3)
    cross_product1 = vec1.cross(vec2)
    cross_product2 = vec2.cross(vec1)

    assert np.allclose(cross_product1.data, -cross_product2.data), "Error: swapping operands must flip the sign"
