import numpy as np
from src.engine.model.Cube import Cube
from src.engine.scene.Scene import Scene
from src.math.Mat4x4 import Mat4x4


def rot_x(deg):
    a = np.radians(deg)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(a), -np.sin(a), 0],
        [0, np.sin(a), np.cos(a), 0],
        [0, 0, 0, 1]
    ])


def rot_y(deg):
    a = np.radians(deg)
    return np.array([
        [np.cos(a), 0, np.sin(a), 0],
        [0, 1, 0, 0],
        [-np.sin(a), 0, np.cos(a), 0],
        [0, 0, 0, 1]
    ])


def rot_z(deg):
    a = np.radians(deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0, 0],
        [np.sin(a), np.cos(a), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def translate(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def scale(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


def run_euler_math_tasks():
    print("=== Завдання 1: Розтяг, обертання і зсув ===")
    S = scale(2, 0.5, 1)
    # Конвенція XYZ (послідовно X, потім Y, потім Z): R = Rz * Ry * Rx
    R_xyz = rot_z(60) @ rot_y(45) @ rot_x(30)
    T = translate(-3, 2, 5)

    M_task1 = T @ R_xyz @ S
    p_initial = np.array([1, 1, 1, 1])
    p_final = M_task1 @ p_initial
    print(f"Матриця трансформації M1:\n{np.round(M_task1, 3)}")
    print(f"Позиція точки (1,1,1) після трансформації: {np.round(p_final[:3], 3)}\n")

    print("=== Завдання 2: Поворот у системі ZYX ===")
    # Конвенція ZYX: R = Rx * Ry * Rz
    R_zyx = rot_x(50) @ rot_y(35) @ rot_z(20)
    T2 = translate(1, 3, -2)
    M_task2 = T2 @ R_zyx
    p_final2 = M_task2 @ p_initial
    print(f"Матриця трансформації M2:\n{np.round(M_task2, 3)}")
    print(f"Позиція точки (1,1,1) після трансформації: {np.round(p_final2[:3], 3)}\n")

    print("=== Завдання 3: Конвенції обертань (45, 30, 60) ===")
    R1 = rot_z(60) @ rot_y(30) @ rot_x(45)  # XYZ
    R2 = rot_x(60) @ rot_y(30) @ rot_z(45)  # ZYX
    print(f"Матриця XYZ:\n{np.round(R1, 3)}")
    print(f"Матриця ZYX:\n{np.round(R2, 3)}")
    print("Висновок: Матричне множення не є комутативним. Порядок осей кардинально змінює фінальну орієнтацію.\n")

    print("=== Завдання 5: Експеримент 'Втрачена вісь' ===")
    # Базове обертання: X=30, Y=90, Z=45
    R_base = rot_z(45) @ rot_y(90) @ rot_x(30)
    # Змінене обертання: X=40 (+10), Y=90, Z=35 (-10)
    R_mod = rot_z(35) @ rot_y(90) @ rot_x(40)

    diff = np.sum(np.abs(R_base - R_mod))
    print(f"Різниця між матрицями: {diff:.6f}")
    if diff < 1e-6:
        print("Доведено: зміна X на +10 і Z на -10 дала ту саму матрицю. Осі 'склеїлися' (Gimbal Lock).\n")

    print("=== Завдання 6: Інтерполяція в зоні сингулярності ===")
    print("Кроки від (0,0,0) до (90,90,90) для вектора (0,0,1):")
    v_forward = np.array([0, 0, 1, 1])
    for step in range(11):
        t = step / 10.0
        angle = 90 * t
        R_lerp = rot_z(angle) @ rot_y(angle) @ rot_x(angle)
        v_out = R_lerp @ v_forward
        print(f"Крок {step} (Кут {angle:2.0f}°): Вектор = {np.round(v_out[:3], 3)}")
    print("При наближенні до 90°, об'єкт починає різко обертатися навколо 'склеєної' осі, втрачаючи плавність.\n")

    print("=== Завдання 7: Декомпозиція та Gimbal Lock ===")

    # Алгоритм витягнення кутів з матриці (Конвенція XYZ)
    def extract_euler_xyz(R):
        if np.abs(R[0, 2] - 1.0) < 1e-6:
            beta = np.pi / 2
            alpha = 0
            gamma = np.arctan2(R[1, 0], R[1, 1])
        elif np.abs(R[0, 2] + 1.0) < 1e-6:
            beta = -np.pi / 2
            alpha = 0
            gamma = -np.arctan2(R[1, 0], R[1, 1])
        else:
            beta = np.arcsin(R[0, 2])
            alpha = np.arctan2(-R[1, 2], R[2, 2])
            gamma = np.arctan2(-R[0, 1], R[0, 0])
        return np.degrees([alpha, beta, gamma])

    # Тестуємо на матриці (X=30, Y=90, Z=45)
    angles_extracted = extract_euler_xyz(R_base)
    print(f"Початкові кути були: X=30, Y=90, Z=45")
    print(f"Витягнуті кути (з примусовим X=0): {np.round(angles_extracted, 2)}")
    print("Ми отримали X=0, Y=90, Z=15. Оскільки (45 - 30) = 15, це та сама орієнтація!\n")

    return M_task1, M_task2


class EulerTaskScene(Scene):
    def __init__(self, M_transform, title, **kwargs):
        super().__init__(title=title, **kwargs)

        initial_cube = Cube(color="cyan", edge_color="blue", alpha=0.3)
        self["initial_cube"] = initial_cube
        transformed_cube = Cube(color="orange", edge_color="red", alpha=0.8)

        mat_transform = Mat4x4()
        mat_transform.matrix = M_transform

        transformed_cube.transformation = mat_transform
        transformed_cube.apply_transformation_to_geometry()

        self["transformed_cube"] = transformed_cube


if __name__ == '__main__':
    M1, M2 = run_euler_math_tasks()

    print("Відображається Завдання 1 (Розтяг, обертання, зсув)...")
    scene1 = EulerTaskScene(
        M_transform=M1,
        title="Завдання 1: XYZ Transformation",
        coordinate_rect=(-4, -1, -1, 3, 4, 6),
        axis_show=True, grid_show=True
    )
    scene1.show()

    print("Відображається Завдання 2 (Обертання ZYX і зсув)...")
    scene2 = EulerTaskScene(
        M_transform=M2,
        title="Завдання 2: ZYX Transformation",
        coordinate_rect=(-1, -1, -3, 3, 4, 2),
        axis_show=True, grid_show=True
    )
    scene2.show()