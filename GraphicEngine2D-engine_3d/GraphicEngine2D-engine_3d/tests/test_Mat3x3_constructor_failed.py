import pytest
import numpy as np
from src.math.Mat3x3 import Mat3x3
from src.math.Vec3 import Vec3


# 1. Передано рядок (expected ValueError)
def test_invalid_matrix_from_string():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3("invalid")

# 2. Передано менше ніж 4 числа (expected ValueError)
def test_invalid_matrix_from_few_numbers():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3(1, 2, 3)  # Expected error

# 3. Передано більше ніж 9 чисел (expected ValueError)
def test_invalid_matrix_from_too_many_numbers():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3(*(range(10)))  # Expected error

# 4. Передано список неправильного розміру (наприклад, 1x4)
def test_invalid_matrix_from_wrong_list_size():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3([[1, 2, 3, 4]])

# 5. Передано 1D `numpy.ndarray` замість 2D (expected ValueError)
def test_invalid_matrix_from_1D_numpy():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3(np.array([1, 2, 3]))  # Expected error

# 6. Передано `numpy.ndarray` неправильної форми (наприклад, 4x4)
def test_invalid_matrix_from_wrong_numpy_shape():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))  # 4x4

# 7. Передано `None` (expected ValueError)
def test_invalid_matrix_from_none():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3(None)  # Pass None

# 8. Передано множину (`set`) замість списку або `numpy.ndarray`
def test_invalid_matrix_from_set():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3({1, 2, 3})  # Set is not supported

# 9. Передано словник (`dict`) замість списку або `numpy.ndarray`
def test_invalid_matrix_from_dict():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3({"a": 1, "b": 2})  # Dict is not a valid type

# 10. Передано булеве значення (`bool`) замість списку або масиву
def test_invalid_matrix_from_bool():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3(True)  # Boolean is not a valid type

# 14. Test помилки при створенні з неправильного списку списків
def test_invalid_matrix_from_irregular_list():
    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3([[1, 2], [3]])  # Invalid nested list

# 15.
def test_matrix_invalid_vec3():
    row1 = Vec3(1, 2, 3)
    row2 = Vec3(4, 5, 6)
    row3 = [7, 8, 9]  # Not Vec3

    with pytest.raises(ValueError, match=Mat3x3.ERROR_MESSAGE_CONSTRUCTOR):
        Mat3x3(row1, row2, row3)

# 16. Перевірка, що некоректний розмір вхідного списку викликає помилку
def test_matrix_invalid_size():
    with pytest.raises(ValueError):
        Mat3x3([[1, 2, 3], [4, 5]])

# 17. Перевірка, що передача неправильного типу викликає помилку
def test_matrix_invalid_type():
    with pytest.raises(ValueError):
        Mat3x3("invalid input")