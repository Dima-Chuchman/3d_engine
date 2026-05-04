import numpy as np


def rotation_matrix_x(phi):
    """
    Builds the rotation matrix around the X axis by the given angle.

    Parameters:
    theta (float): Rotation angle in radians.

    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    cos_phi, sin_phi =  np.cos(phi), np.sin(phi)
    return np.array([
        [1,         0,            0, ],
        [0,    cos_phi,    -sin_phi, ],
        [0,    sin_phi,     cos_phi, ],
    ])


def rotation_matrix_y(phi):
    """
    Builds the rotation matrix around the Y axis by the given angle.

    Parameters:
    theta (float): Rotation angle in radians.

    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    cos_phi, sin_phi =  np.cos(phi), np.sin(phi)
    return np.array([
        [ cos_phi,     0,     sin_phi ],
        [       0,     1,           0 ],
        [-sin_phi,     0,     cos_phi ],
    ])

def rotation_matrix_z(phi):
    """
    Builds the rotation matrix around the Z axis by the given angle.

    Parameters:
    theta (float): Rotation angle in radians.

    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    cos_phi, sin_phi =  np.cos(phi), np.sin(phi)

    return np.array([
        [cos_phi,   -sin_phi,     0 ],
        [sin_phi,    cos_phi,     0 ],
        [      0,          0,     1 ],
    ])


# Usage example:
if __name__ == "__main__":
    euler_angles_45_45_30 = [45, 15, 30]
    x = np.radians(euler_angles_45_45_30[0])  # Angle in degrees is converted to radians
    y = np.radians(euler_angles_45_45_30[1])  # Angle in degrees is converted to radians
    z = np.radians(euler_angles_45_45_30[2])  # Angle in degrees is converted to radians

    Rx = rotation_matrix_x(x)
    Ry = rotation_matrix_y(y)
    Rz = rotation_matrix_z(z)

    print("\nRotation matrix for sequential Euler angles:")
    print()
    print(Rx @ Ry @ Rz)

