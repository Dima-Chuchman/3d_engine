import numpy as np
import random

from src.engine.model.SimplePolygon import SimplePolygon
from src.engine.scene.Scene import Scene


def apply_transformation_3d(matrix, points):
    result_coords = []
    for p in points:
        vec = np.array([p[0], p[1], p[2], 1])
        new_vec = matrix.dot(vec)
        result_coords.extend([float(new_vec[0]), float(new_vec[1]), float(new_vec[2])])
    return result_coords


def T_mat3d(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ], dtype=float)


def S_mat3d(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def R_arbitrary(axis, angle_deg):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    axis = np.array(axis) / np.linalg.norm(axis)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s, 0],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c), 0],
        [0, 0, 0, 1]
    ], dtype=float)


def R_euler(rot_x, rot_y, rot_z, order='XYZ'):
    rad_x, rad_y, rad_z = np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
    Rx = np.array(
        [[1, 0, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x), 0], [0, np.sin(rad_x), np.cos(rad_x), 0], [0, 0, 0, 1]])
    Ry = np.array(
        [[np.cos(rad_y), 0, np.sin(rad_y), 0], [0, 1, 0, 0], [-np.sin(rad_y), 0, np.cos(rad_y), 0], [0, 0, 0, 1]])
    Rz = np.array(
        [[np.cos(rad_z), -np.sin(rad_z), 0, 0], [np.sin(rad_z), np.cos(rad_z), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    if order == 'XYZ':
        return Rz @ Ry @ Rx
    elif order == 'ZYX':
        return Rx @ Ry @ Rz
    return np.eye(4)


CUBE_VERTICES = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
]
TETRAHEDRON_VERTICES = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
TRIANGLE_VERTICES = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
RECTANGLE_VERTICES = [(1, 2, 0), (4, 2, 0), (4, 5, 0), (1, 5, 0)]


# ЗАВДАННЯ 1-10: Сцени для візуалізації

class Task1Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cube_init = SimplePolygon(color="blue", line_style="--")
        cube_init.set_geometry(*[c for p in CUBE_VERTICES for c in p])
        self["init"] = cube_init

        R = R_arbitrary((1, 1, 0), 45)
        T = T_mat3d(2, -1, 3)
        M_total = T @ R

        cube_final = SimplePolygon(color="red")
        cube_final.set_geometry(*apply_transformation_3d(M_total, CUBE_VERTICES))
        self["final"] = cube_final


class Task2Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cube_init = SimplePolygon(color="blue", line_style="--")
        cube_init.set_geometry(*[c for p in CUBE_VERTICES for c in p])
        self["init"] = cube_init

        S = S_mat3d(2, 0.5, 1)
        R = R_euler(30, 45, 60, order='XYZ')
        T = T_mat3d(-3, 2, 5)
        M_total = T @ R @ S

        cube_final = SimplePolygon(color="green")
        cube_final.set_geometry(*apply_transformation_3d(M_total, CUBE_VERTICES))
        self["final"] = cube_final


class Task3Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cube_init = SimplePolygon(color="blue", line_style="--")
        cube_init.set_geometry(*[c for p in CUBE_VERTICES for c in p])
        self["init"] = cube_init

        R_z = R_arbitrary((0, 0, 1), 60)
        R_diag = R_arbitrary((1, 1, 1), 45)
        T = T_mat3d(4, -2, 1)
        M_total = T @ R_diag @ R_z

        cube_final = SimplePolygon(color="purple")
        cube_final.set_geometry(*apply_transformation_3d(M_total, CUBE_VERTICES))
        self["final"] = cube_final


class Task4Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cube_init = SimplePolygon(color="blue", line_style="--")
        cube_init.set_geometry(*[c for p in CUBE_VERTICES for c in p])
        self["init"] = cube_init

        R = R_euler(20, 35, 50, order='ZYX')
        T = T_mat3d(1, 3, -2)
        M_total = T @ R

        cube_final = SimplePolygon(color="orange")
        cube_final.set_geometry(*apply_transformation_3d(M_total, CUBE_VERTICES))
        self["final"] = cube_final


class Task5Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tetra_init = SimplePolygon(color="blue", line_style="--")
        tetra_init.set_geometry(*[c for p in TETRAHEDRON_VERTICES for c in p])
        self["init"] = tetra_init

        angle = random.uniform(10, 90)
        axis = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        dx, dy, dz = random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)

        R = R_arbitrary(axis, angle)
        T = T_mat3d(dx, dy, dz)
        M_total = T @ R

        tetra_final = SimplePolygon(color="cyan")
        tetra_final.set_geometry(*apply_transformation_3d(M_total, TETRAHEDRON_VERTICES))
        self["final"] = tetra_final


class Task6Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cube_init = SimplePolygon(color="blue", line_style="--")
        cube_init.set_geometry(*[c for p in CUBE_VERTICES for c in p])
        self["init"] = cube_init

        # Обертання навколо опорної точки (2, 0, 3)
        T_pivot_to_origin = T_mat3d(-2, 0, -3)
        R_y = R_arbitrary((0, 1, 0), 45)
        T_pivot_back = T_mat3d(2, 0, 3)

        T_final = T_mat3d(-1, 2, 4)

        M_total = T_final @ T_pivot_back @ R_y @ T_pivot_to_origin

        cube_final = SimplePolygon(color="magenta")
        cube_final.set_geometry(*apply_transformation_3d(M_total, CUBE_VERTICES))
        self["final"] = cube_final


class Task7Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cube_init = SimplePolygon(color="blue", line_style="--")
        cube_init.set_geometry(*[c for p in CUBE_VERTICES for c in p])
        self["init"] = cube_init

        pivot = (1, 2, 3)
        T_to_origin = T_mat3d(-pivot[0], -pivot[1], -pivot[2])
        T_back = T_mat3d(pivot[0], pivot[1], pivot[2])

        S_x = S_mat3d(3, 1, 1)
        R_z = R_arbitrary((0, 0, 1), 30)

        # Композиція з опорною точкою
        M_scale = T_back @ S_x @ T_to_origin
        M_rot = T_back @ R_z @ T_to_origin
        M_total = M_rot @ M_scale

        cube_final = SimplePolygon(color="yellow")
        cube_final.set_geometry(*apply_transformation_3d(M_total, CUBE_VERTICES))
        self["final"] = cube_final


class Task8Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tri_init = SimplePolygon(color="blue", line_style="--")
        tri_init.set_geometry(*[c for p in TRIANGLE_VERTICES for c in p])
        self["init"] = tri_init

        pivot = (2, 3, 4)
        T_to_origin = T_mat3d(-pivot[0], -pivot[1], -pivot[2])
        R_axis = R_arbitrary((1, 1, 1), 90)
        T_back = T_mat3d(pivot[0], pivot[1], pivot[2])

        T_final = T_mat3d(0, -3, 2)
        M_total = T_final @ T_back @ R_axis @ T_to_origin

        tri_final = SimplePolygon(color="pink")
        tri_final.set_geometry(*apply_transformation_3d(M_total, TRIANGLE_VERTICES))
        self["final"] = tri_final


class Task9Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rect_init = SimplePolygon(color="blue", line_style="--")
        rect_init.set_geometry(*[c for p in RECTANGLE_VERTICES for c in p])
        self["init"] = rect_init

        pivot = (3, 3, 0)
        T_to_origin = T_mat3d(-pivot[0], -pivot[1], -pivot[2])
        T_back = T_mat3d(pivot[0], pivot[1], pivot[2])

        R_y = R_arbitrary((0, 1, 0), 60)
        R_x = R_arbitrary((1, 0, 0), 30)

        M_total = T_back @ R_x @ R_y @ T_to_origin

        rect_final = SimplePolygon(color="brown")
        rect_final.set_geometry(*apply_transformation_3d(M_total, RECTANGLE_VERTICES))
        self["final"] = rect_final


class Task10Scene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cube_init = SimplePolygon(color="blue", line_style="--")
        cube_init.set_geometry(*[c for p in CUBE_VERTICES for c in p])
        self["init"] = cube_init

        pivot = (1, 1, 1)
        T_to_origin = T_mat3d(-pivot[0], -pivot[1], -pivot[2])
        T_back = T_mat3d(pivot[0], pivot[1], pivot[2])

        S_x = S_mat3d(2, 1, 1)
        R_y = R_arbitrary((0, 1, 0), 45)
        T_final = T_mat3d(-3, 4, 2)

        M_total = T_final @ T_back @ R_y @ S_x @ T_to_origin

        cube_final = SimplePolygon(color="olive")
        cube_final.set_geometry(*apply_transformation_3d(M_total, CUBE_VERTICES))
        self["final"] = cube_final


# ЗАВДАННЯ 11-15: Аналітичні та математичні функції

def task_11_compare_rotations():
    print("--- Завдання 11 ---")
    # Трансформація А (Зовнішня - Світові осі, порядок множення зліва направо: Rz * Ry * Rx)
    R_x = R_arbitrary((1, 0, 0), 30)
    R_y = R_arbitrary((0, 1, 0), 45)
    R_z = R_arbitrary((0, 0, 1), 60)
    M_A = R_z @ R_y @ R_x

    # Трансформація Б (Внутрішня - Локальні осі, порядок ZYX, множення зліва направо: Rx * Ry * Rz)
    R_z_int = R_arbitrary((0, 0, 1), 60)
    R_y_int = R_arbitrary((0, 1, 0), 45)
    R_x_int = R_arbitrary((1, 0, 0), 30)
    M_B = R_x_int @ R_y_int @ R_z_int  # Для внутрішніх матриці множаться справа

    print("Матриця A (Зовнішні):\n", np.round(M_A, 4))
    print("Матриця B (Внутрішні):\n", np.round(M_B, 4))
    print("Чи ідентичні? ", np.allclose(M_A, M_B))


def task_12_decomposition(M):
    print("--- Завдання 12 ---")
    M = np.array(M)
    # 1. Вектор перенесення
    T = M[:3, 3]

    # 2. Масштаб
    RS = M[:3, :3]
    sx = np.linalg.norm(RS[:, 0])
    sy = np.linalg.norm(RS[:, 1])
    sz = np.linalg.norm(RS[:, 2])
    S = np.array([sx, sy, sz])

    # 3. Чиста матриця обертання
    R = RS / S
    ortho_check = np.allclose(R.T @ R, np.eye(3))

    # 4. Кут і вісь
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    angle_deg = np.degrees(angle)

    antisymmetric = (R - R.T) / 2
    axis = np.array([antisymmetric[2, 1], antisymmetric[0, 2], antisymmetric[1, 0]])
    if np.linalg.norm(axis) > 0:
        axis = axis / np.linalg.norm(axis)

    print(f"Вектор перенесення T: {T}")
    print(f"Масштаб S: {S}")
    print(f"Ортогональність R: {ortho_check}")
    print(f"Кут повороту: {angle_deg:.2f} градусів")
    print(f"Вісь повороту: {axis}")


def task_13_internal_rotations():
    print("--- Завдання 13 ---")
    # Внутрішні трансформації (множаться справа)
    M1 = R_arbitrary((1, 0, 0), 45)
    M2 = T_mat3d(0, 2, 0)
    M3 = R_arbitrary((0, 0, 1), 30)

    M_total = M1 @ M2 @ M3
    points = apply_transformation_3d(M_total, TETRAHEDRON_VERTICES)
    print("Фінальні координати (спліщеним масивом):", np.round(points, 2))


def task_15_complex_decomposition():
    print("--- Завдання 15 ---")
    pivot = (1, 1, 1)
    T_to_origin = T_mat3d(-pivot[0], -pivot[1], -pivot[2])
    T_back = T_mat3d(pivot[0], pivot[1], pivot[2])

    S_all = S_mat3d(2, 2, 2)
    M_scale = T_back @ S_all @ T_to_origin

    # Внутрішнє обертання Y на 90 (множиться справа)
    R_y_int = R_arbitrary((0, 1, 0), 90)

    # Зовнішнє переміщення (множиться зліва)
    T_ext = T_mat3d(-3, 4, 2)

    M_total = T_ext @ M_scale @ R_y_int
    print("Фінальна матриця завдання 15:\n", np.round(M_total, 4))

    task_12_decomposition(M_total)


if __name__ == '__main__':
    rect = (-8, -8, -8, 8, 8, 8)

    # ЗАПУСК СЦЕН (Завдання 1-10)
    Task1Scene(coordinate_rect=rect, title="Завдання 1", grid_show=True, axis_show=True).show()
    Task2Scene(coordinate_rect=rect, title="Завдання 2", grid_show=True, axis_show=True).show()
    Task3Scene(coordinate_rect=rect, title="Завдання 3", grid_show=True, axis_show=True).show()
    Task4Scene(coordinate_rect=rect, title="Завдання 4", grid_show=True, axis_show=True).show()
    Task5Scene(coordinate_rect=rect, title="Завдання 5", grid_show=True, axis_show=True).show()
    Task6Scene(coordinate_rect=rect, title="Завдання 6", grid_show=True, axis_show=True).show()
    Task7Scene(coordinate_rect=rect, title="Завдання 7", grid_show=True, axis_show=True).show()
    Task8Scene(coordinate_rect=rect, title="Завдання 8", grid_show=True, axis_show=True).show()
    Task9Scene(coordinate_rect=rect, title="Завдання 9", grid_show=True, axis_show=True).show()
    Task10Scene(coordinate_rect=rect, title="Завдання 10", grid_show=True, axis_show=True).show()

    task_11_compare_rotations()

    M_task12 = [
        [0.707, -0.707, 0, 5],
        [1.060, 1.060, 1.0, 2],
        [0.707, 0.707, 1.414, 3],
        [0, 0, 0, 1]
    ]
    task_12_decomposition(M_task12)

    task_13_internal_rotations()

    task_15_complex_decomposition()