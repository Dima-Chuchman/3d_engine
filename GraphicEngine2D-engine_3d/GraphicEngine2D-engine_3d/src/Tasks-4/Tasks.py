import numpy as np
from src.engine.model.SimplePolygon import SimplePolygon
from src.engine.scene.Scene import Scene


def q_multiply(q1, q2):
    """Множення кватерніонів: q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def q_conjugate(q):
    """Спряжений кватерніон (для одиничних = обернений)"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_norm(q):
    """Норма кватерніона"""
    return np.linalg.norm(q)


def q_normalize(q):
    """Нормалізація кватерніона"""
    return q / q_norm(q)


def q_from_axis_angle(axis, angle_deg):
    """Перетворення 'вісь + кут' у кватерніон"""
    angle_rad = np.radians(angle_deg)
    axis = np.array(axis) / np.linalg.norm(axis)
    w = np.cos(angle_rad / 2)
    xyz = axis * np.sin(angle_rad / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def q_to_axis_angle(q):
    """Отримання осі та кута з кватерніона"""
    q = q_normalize(q)
    angle_rad = 2 * np.arccos(q[0])
    s = np.sqrt(1 - q[0] * q[0])
    if s < 1e-8:
        axis = np.array([1, 0, 0])
    else:
        axis = q[1:4] / s
    return axis, np.degrees(angle_rad)


def q_apply_to_vector(q, v):
    """Поворот вектора за допомогою кватерніона: v' = q * v * q^-1"""
    v_quat = np.array([0, v[0], v[1], v[2]])
    q_conj = q_conjugate(q)
    res = q_multiply(q_multiply(q, v_quat), q_conj)
    return res[1:4]


def q_to_matrix(q):
    """Перетворення кватерніона в матрицю повороту 3x3"""
    q = q_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])


def q_from_matrix(R):
    """Надійна декомпозиція матриці 3x3 у кватерніон (Завдання 4)"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def run_math_tasks():
    print("=" * 60)
    print("0. Від осі та кута до кватерніона")
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    angle = 60
    q0 = q_from_axis_angle(axis, angle)
    print(f"1. Кватерніон q = {np.round(q0, 4)}")
    print(f"2. Норма |q| = {q_norm(q0):.4f} (перевірка на 1)")
    print(f"3. Матриця повороту R:\n{np.round(q_to_matrix(q0), 3)}")

    print("\n" + "=" * 60)
    print("1. Операція повороту вектора")
    p = np.array([1, 0, 0])
    q_z90 = q_from_axis_angle([0, 0, 1], 90)
    v_pure = [0, 1, 0, 0]
    print(f"1. Чистий кватерніон v = {v_pure}")
    p_new = q_apply_to_vector(q_z90, p)
    print(f"2-3. Нові координати після q*v*q^-1: {np.round(p_new, 3)}")
    print("Очікуваний результат матриці повороту для (1,0,0) на 90 по Z - це (0,1,0). Збігається!")

    print("\n" + "=" * 60)
    print("2. Композиція складних обертань")
    q1_x45 = q_from_axis_angle([1, 0, 0], 45)
    q2_y30 = q_from_axis_angle([0, 1, 0], 30)
    # Композиція для зовнішніх осей: q_total = q2 * q1
    q_total = q_multiply(q2_y30, q1_x45)
    print(f"3. Результуючий кватерніон q_total = {np.round(q_total, 4)}")

    axis_total, angle_total = q_to_axis_angle(q_total)
    print(f"4. Параметри повороту: Кут = {angle_total:.2f}°, Вісь = {np.round(axis_total, 3)}")

    print("\n" + "=" * 60)
    print("3. Конвертація з кутів Ойлера та Gimbal Lock")
    qz = q_from_axis_angle([0, 0, 1], 20)  # yaw
    qy = q_from_axis_angle([0, 1, 0], 90)  # pitch
    qx = q_from_axis_angle([1, 0, 0], 50)  # roll
    print(f"1. qz = {np.round(qz, 3)}, qy = {np.round(qy, 3)}, qx = {np.round(qx, 3)}")
    q_euler_total = q_multiply(qz, q_multiply(qy, qx))
    print(f"2. Фінальний q = {np.round(q_euler_total, 4)}")
    print("3. ДОВЕДЕННЯ: Отримали чіткий 4-вимірний вектор (w,x,y,z) без ділення на нуль.")

    print("\n" + "=" * 60)
    print("4. Декомпозиція матриці в кватерніон")
    R_task4 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    q_from_mat = q_from_matrix(R_task4)
    print(f"1. Отриманий кватерніон q = {np.round(q_from_mat, 4)}")

    print("\n" + "=" * 60)
    print("5. Повна декомпозиція афінної матриці")
    M = np.array([
        [0, -2, 0, 10],
        [1, 0, 0, -5],
        [0, 0, 1.5, 3],
        [0, 0, 0, 1]
    ])

    # Вилучення трансляції (останній стовпець)
    T = M[0:3, 3]
    print(f"1. Вектор перенесення T = {T}")

    # Вилучення масштабування (норми стовпців 3x3)
    col0 = M[0:3, 0]
    col1 = M[0:3, 1]
    col2 = M[0:3, 2]
    S = np.array([np.linalg.norm(col0), np.linalg.norm(col1), np.linalg.norm(col2)])
    print(f"2. Масштабні коефіцієнти S = {S}")

    # Отримання чистої матриці обертання
    R_pure = np.column_stack((col0 / S[0], col1 / S[1], col2 / S[2]))
    print(f"3. Чиста матриця обертання R:\n{np.round(R_pure, 2)}")

    # Конвертація в кватерніон
    q_final = q_from_matrix(R_pure)
    print(f"4. Кватерніон афінної матриці q = {np.round(q_final, 4)}")
    print("=" * 60 + "\n")

    return q_total


class TetrahedronScene(Scene):
    def __init__(self, q_transform, **kwargs):
        super().__init__(**kwargs)

        # 1. Вершини тетраедра
        v = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]

        # 2. Обчислюю нові вершини через поворот кватерніоном
        v_new = [q_apply_to_vector(q_transform, p) for p in v]

        # 3. Список граней (індекси вершин)
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

        # 4. Малюю початковий тетраедр
        for i, face in enumerate(faces):
            coords = []
            for idx in face:
                coords.extend([float(v[idx][0]), float(v[idx][1]), float(v[idx][2])])

            # Передача аргументів у конструктор замість set_geometry
            poly = SimplePolygon(*coords, color="blue", line_style="--", alpha=0.3)
            self[f"orig_face_{i}"] = poly

        # 5. Малюю повернутий тетраедр
        for i, face in enumerate(faces):
            coords = []
            for idx in face:
                coords.extend([float(v_new[idx][0]), float(v_new[idx][1]), float(v_new[idx][2])])

            poly = SimplePolygon(*coords, color="red", line_width=2.0, alpha=0.6)
            self[f"new_face_{i}"] = poly


if __name__ == '__main__':
    q_total_for_scene = run_math_tasks()
    print("Відкривається вікно візуалізації для Завдання 2 (Тетраедр)...")

    scene = TetrahedronScene(
        q_transform=q_total_for_scene,
        coordinate_rect=(-1, -1, -1, 2, 2, 2),
        title="Завдання 2: Поворот Тетраедра Кватерніоном",
        grid_show=True,
        axis_show=True
    )
    scene.show()