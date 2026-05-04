import pytest
import numpy as np
from src.math.Vec3 import Vec3  # Make sure the import path is correct!

# 1. Create zero vector
def test_default_vector():
    vec = Vec3()
    expected = np.zeros(3, dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be zero vector"

# 2. Create from 3 numbers
def test_vector_from_three_numbers():
    vec = Vec3(1, 2, 3)
    expected = np.array([1, 2, 3], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (1, 2, 3)"

# 2.1. Create from 2 numbers
def test_vector_from_two_numbers():
    vec = Vec3(1, 2)
    expected = np.array([1, 2, 0], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (1, 2, 0)"

# 3. Create from 3-element list
def test_vector_from_list():
    vec = Vec3([1, 2, 3])
    expected = np.array([1, 2, 3], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (1, 2, 3) from list"

# 4. Create from 3-element numpy.ndarray
def test_vector_from_numpy():
    data = np.array([1, 2, 3], dtype=float)
    vec = Vec3(data)
    assert np.allclose(vec.data, data), "Error: should be vector (1, 2, 3) from numpy"

# 5. Create a copy of another Vec3
def test_vector_from_existing_vec3():
    original = Vec3(4, 5, 6)
    copy = Vec3(original)
    assert np.allclose(copy.data, original.data), "Vector copy must be identical to original"

# 6. Create from tuple
def test_vector_from_tuple():
    vec = Vec3((7, 8, 9))
    expected = np.array([7, 8, 9], dtype=float)
    assert np.allclose(vec.data, expected), "Error: should be vector (7, 8, 9) from tuple"

# 7. Invalid size (e.g., list of 2 elements) -> expected ValueError
def test_vector_invalid_size():
    with pytest.raises(ValueError, match=Vec3.ERROR_MESSAGE_CONSTRUCTOR):
        Vec3([1, 2])  # Wrong size

# 8. Invalid type (e.g., string) -> expected TypeError
def test_vector_invalid_type():
    with pytest.raises(ValueError, match=Vec3.ERROR_MESSAGE_CONSTRUCTOR):
        Vec3("invalid input")  # Wrong type

# 9. Create from numpy array of wrong size -> expected ValueError
def test_vector_invalid_numpy_size():
    with pytest.raises(ValueError, match=Vec3.ERROR_MESSAGE_CONSTRUCTOR):
        Vec3(np.array([1, 2, 3, 4]))  # Too many elements

# 10. Create from unsupported type (not Vec3, list, tuple or numpy.ndarray) -> expected TypeError
def test_vector_invalid_object():
    class Dummy:
        pass

    with pytest.raises(ValueError, match=Vec3.ERROR_MESSAGE_CONSTRUCTOR):
        Vec3(Dummy())  # Invalid type
