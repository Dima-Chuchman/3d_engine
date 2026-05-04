import numpy as np

from src.math.Mat3x3 import Mat3x3
from src.math.Vec3 import Vec3


# 1. Перевірка створення одиничної матриці без аргументів
def test_identity_matrix():
    mat = Mat3x3()
    expected = np.eye(3, dtype=float)
    assert np.allclose(mat.data, expected)

# 2. Перевірка створення матриці з 9 чисел (розгорнутий формат)
def test_matrix_from_9_elements():
    mat = Mat3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    assert np.allclose(mat.data, expected)

# 3. Перевірка створення матриці з 4 чисел (2x2 розширена до 3x3)
def test_matrix_from_4_elements():
    mat = Mat3x3(1, 2, 3, 4)
    expected = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected)

# 4. Перевірка створення з 3x3 списку
def test_matrix_from_list_3x3():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mat = Mat3x3(data)
    expected = np.array(data, dtype=float)
    assert np.allclose(mat.data, expected)

# 5. Перевірка створення з 2x2 списку (автоматичне доповнення до 3x3)
def test_matrix_from_list_2x2():
    data = [[1, 2], [3, 4]]
    mat = Mat3x3(data)
    expected = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected)

# 6. Перевірка створення копії матриці (конструктор з Mat3x3)
def test_matrix_from_existing():
    original = Mat3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    copy = Mat3x3(original)
    assert np.allclose(copy.data, original.data)

# 7. Перевірка створення з numpy-масиву 3x3
def test_matrix_from_numpy_array():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    mat = Mat3x3(data)
    assert np.allclose(mat.data, data)

# 8. Перевірка створення з numpy-масиву 2x2 (розширення до 3x3)
def test_matrix_from_numpy_2x2():
    data = np.array([[1, 2], [3, 4]], dtype=float)
    mat = Mat3x3(data)
    expected = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(mat.data, expected)

def test_matrix_from_three_vec3():
    row1 = Vec3(1, 2, 3)
    row2 = Vec3(4, 5, 6)
    row3 = Vec3(7, 8, 9)

    mat = Mat3x3(row1, row2, row3)
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

    assert np.allclose(mat.data, expected), "Error creating matrix from three Vec3"



