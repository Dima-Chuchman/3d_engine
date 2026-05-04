import numpy as np
import pytest

from src.math.Mat4x4 import Mat4x4
from src.math.Quaternion import Quaternion
from src.math.Vec3 import Vec3
from src.math.utils_quat import (
    euler_to_quaternion,
    euler_xyz_to_quaternion,
    quaternion_to_euler,
    quaternion_to_euler_xyz,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    angle_axis_to_quaternion,
    quaternion_to_angle_axis,
    slerp,
    nlerp,
    quaternion_dot,
    are_same_rotation,
    angle_between_rotations,
    quaternion_identity,
    is_unit_quaternion,
)

ALL_CONFIGS = [
    "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
    "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ",
]


# ══════════════════════════════════════════════════════════════════════════════
# Euler Angles → Quaternion
# ══════════════════════════════════════════════════════════════════════════════

class TestEulerToQuaternion:

    def test_xyz_direct_vs_generic(self):
        """euler_xyz_to_quaternion and euler_to_quaternion('XYZ') give identical result."""
        phi, theta, psi = np.radians((30, 45, 60))
        q_direct = euler_xyz_to_quaternion(phi, theta, psi)
        q_generic = euler_to_quaternion(phi, theta, psi, "XYZ")
        assert np.allclose(q_direct.q, q_generic.q, atol=1e-10)

    def test_zero_angles_give_identity(self):
        """Angles (0,0,0) for any configuration give identity quaternion."""
        for cfg in ALL_CONFIGS:
            q = euler_to_quaternion(0.0, 0.0, 0.0, cfg)
            assert np.allclose(q.q, [1, 0, 0, 0], atol=1e-10), \
                f"{cfg}: (0,0,0) should give identity quaternion"

    def test_result_is_unit_quaternion(self):
        """Result of euler_to_quaternion is always a unit quaternion."""
        angles = np.radians((30, 45, 60))
        for cfg in ALL_CONFIGS:
            q = euler_to_quaternion(*angles, cfg)
            assert is_unit_quaternion(q, atol=1e-6), \
                f"{cfg}: result must be a unit quaternion"

    def test_case_insensitive(self):
        """Configuration is case-insensitive."""
        angles = np.radians((20, 35, 50))
        for cfg in ALL_CONFIGS:
            q_upper = euler_to_quaternion(*angles, cfg.upper())
            q_lower = euler_to_quaternion(*angles, cfg.lower())
            assert np.allclose(q_upper.q, q_lower.q, atol=1e-10), \
                f"{cfg}: upper and lower case give different results"

    def test_invalid_configuration(self):
        """Unknown configuration raises ValueError."""
        with pytest.raises(ValueError):
            euler_to_quaternion(0.1, 0.2, 0.3, "ABC")

    def test_xyz_single_axis_x(self):
        """XYZ with theta=psi=0: should match Quaternion.rotation_x."""
        phi = np.radians(45)
        q_expected = Quaternion.rotation_x(phi)
        q = euler_to_quaternion(phi, 0.0, 0.0, "XYZ")
        assert are_same_rotation(q, q_expected)

    def test_xyz_single_axis_y(self):
        """XYZ with phi=psi=0: should match Quaternion.rotation_y."""
        theta = np.radians(45)
        q_expected = Quaternion.rotation_y(theta)
        q = euler_to_quaternion(0.0, theta, 0.0, "XYZ")
        assert are_same_rotation(q, q_expected)

    def test_xyz_single_axis_z(self):
        """XYZ with phi=theta=0: should match Quaternion.rotation_z."""
        psi = np.radians(45)
        q_expected = Quaternion.rotation_z(psi)
        q = euler_to_quaternion(0.0, 0.0, psi, "XYZ")
        assert are_same_rotation(q, q_expected)

    def test_zyx_matches_component_product(self):
        """ZYX: Qz(phi)*Qy(theta)*Qx(psi) = euler_to_quaternion(..., 'ZYX')."""
        phi, theta, psi = np.radians((30, 45, 60))
        q_expected = (Quaternion.rotation_z(phi)
                      * Quaternion.rotation_y(theta)
                      * Quaternion.rotation_x(psi))
        q = euler_to_quaternion(phi, theta, psi, "ZYX")
        assert are_same_rotation(q, q_expected)


# ══════════════════════════════════════════════════════════════════════════════
# Quaternion → Euler Angles (round-trip)
# ══════════════════════════════════════════════════════════════════════════════

class TestQuaternionToEuler:

    def test_xyz_direct_vs_generic(self):
        """quaternion_to_euler_xyz and quaternion_to_euler('XYZ') give identical result."""
        q = euler_xyz_to_quaternion(*np.radians((30, 45, 60)))
        angles_direct = quaternion_to_euler_xyz(q)
        angles_generic = quaternion_to_euler(q, "XYZ")
        assert np.allclose(angles_direct, angles_generic, atol=1e-6)

    def test_roundtrip_all_configs(self):
        """Round-trip euler→quaternion→euler for all configurations."""
        angles_orig = np.radians((30, 45, 60))
        for cfg in ALL_CONFIGS:
            q = euler_to_quaternion(*angles_orig, cfg)
            recovered = quaternion_to_euler(q, cfg)
            assert np.allclose(np.degrees(recovered), np.degrees(angles_orig), atol=1e-5), \
                f"{cfg}: angle round-trip failed"

    def test_roundtrip_various_angles(self):
        """Round-trip for various angles in XYZ."""
        test_angles = [
            (10, 20, 30),
            (-45, 30, 90),
            (90, 0, 0),
            (0, 90, 0),
            (0, 0, 90),
        ]
        for deg in test_angles:
            angles = np.radians(deg)
            q = euler_xyz_to_quaternion(*angles)
            recovered = quaternion_to_euler_xyz(q)
            assert np.allclose(np.degrees(recovered), deg, atol=1e-5), \
                f"Round-trip for {deg} failed"

    def test_identity_quaternion_gives_zero_angles(self):
        """Identity quaternion → (0, 0, 0) for all Tait-Bryan configurations."""
        q = quaternion_identity()
        for cfg in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
            angles = quaternion_to_euler(q, cfg)
            assert np.allclose(angles, (0.0, 0.0, 0.0), atol=1e-10), \
                f"{cfg}: identity quaternion should give (0,0,0)"

    def test_negative_quaternion_same_rotation(self):
        """q and -q represent the same rotation → identical Euler angles."""
        q = euler_xyz_to_quaternion(*np.radians((30, 45, 60)))
        angles_pos = quaternion_to_euler_xyz(q)
        angles_neg = quaternion_to_euler_xyz(-q)
        assert np.allclose(angles_pos, angles_neg, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Rotation Matrix ↔ Quaternion
# ══════════════════════════════════════════════════════════════════════════════

class TestRotationMatrixQuaternion:

    def test_matrix_to_quaternion_and_back(self):
        """rotation_matrix → quaternion → rotation_matrix gives the original matrix."""
        phi, theta, psi = np.radians((30, 45, 60))
        mat = Mat4x4.rotation_euler(phi, theta, psi, "XYZ")
        q = rotation_matrix_to_quaternion(mat)
        mat_recovered = quaternion_to_rotation_matrix(q)
        assert np.allclose(mat.data, mat_recovered.data, atol=1e-6)

    def test_quaternion_to_matrix_and_back(self):
        """quaternion → rotation_matrix → quaternion gives original quaternion (up to sign)."""
        q = euler_to_quaternion(*np.radians((30, 45, 60)), "XYZ")
        mat = quaternion_to_rotation_matrix(q)
        q_recovered = rotation_matrix_to_quaternion(mat)
        assert are_same_rotation(q, q_recovered)

    def test_identity_matrix_gives_identity_quaternion(self):
        """Identity matrix → identity quaternion."""
        q = rotation_matrix_to_quaternion(Mat4x4.identity())
        assert are_same_rotation(q, quaternion_identity())

    def test_rotation_x_90(self):
        """Rx(90°) gives quaternion (cos45°, sin45°, 0, 0)."""
        mat = Mat4x4.rotation_x(np.radians(90))
        q = rotation_matrix_to_quaternion(mat)
        q_expected = Quaternion.rotation_x(np.radians(90))
        assert are_same_rotation(q, q_expected)

    def test_rotation_y_90(self):
        """Ry(90°) gives quaternion (cos45°, 0, sin45°, 0)."""
        mat = Mat4x4.rotation_y(np.radians(90))
        q = rotation_matrix_to_quaternion(mat)
        q_expected = Quaternion.rotation_y(np.radians(90))
        assert are_same_rotation(q, q_expected)

    def test_rotation_z_90(self):
        """Rz(90°) gives quaternion (cos45°, 0, 0, sin45°)."""
        mat = Mat4x4.rotation_z(np.radians(90))
        q = rotation_matrix_to_quaternion(mat)
        q_expected = Quaternion.rotation_z(np.radians(90))
        assert are_same_rotation(q, q_expected)

    def test_non_orthogonal_matrix_raises(self):
        """Non-orthogonal matrix raises ValueError."""
        bad_mat = Mat4x4(2, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1)
        with pytest.raises(ValueError):
            rotation_matrix_to_quaternion(bad_mat)

    def test_result_is_unit_quaternion(self):
        """Result of rotation_matrix_to_quaternion is always a unit quaternion."""
        for angle_deg in [0, 30, 90, 180]:
            mat = Mat4x4.rotation_z(np.radians(angle_deg))
            q = rotation_matrix_to_quaternion(mat)
            assert is_unit_quaternion(q), f"Not a unit quaternion for {angle_deg}°"

    def test_quaternion_to_matrix_is_rotation_matrix(self):
        """quaternion_to_rotation_matrix returns orthogonal matrix with det=1."""
        q = euler_to_quaternion(*np.radians((30, 45, 60)), "ZYX")
        mat = quaternion_to_rotation_matrix(q)
        r = mat.data[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Angle-Axis ↔ Quaternion
# ══════════════════════════════════════════════════════════════════════════════

class TestAngleAxisQuaternion:

    def test_angle_axis_to_quaternion_x(self):
        """angle_axis_to_quaternion for X axis = Quaternion.rotation_x."""
        angle = np.radians(60)
        q = angle_axis_to_quaternion(angle, [1, 0, 0])
        q_expected = Quaternion.rotation_x(angle)
        assert are_same_rotation(q, q_expected)

    def test_angle_axis_to_quaternion_y(self):
        """angle_axis_to_quaternion for Y axis = Quaternion.rotation_y."""
        angle = np.radians(90)
        q = angle_axis_to_quaternion(angle, [0, 1, 0])
        q_expected = Quaternion.rotation_y(angle)
        assert are_same_rotation(q, q_expected)

    def test_angle_axis_to_quaternion_z(self):
        """angle_axis_to_quaternion for Z axis = Quaternion.rotation_z."""
        angle = np.radians(45)
        q = angle_axis_to_quaternion(angle, [0, 0, 1])
        q_expected = Quaternion.rotation_z(angle)
        assert are_same_rotation(q, q_expected)

    def test_roundtrip_angle_axis(self):
        """angle_axis → quaternion → angle_axis gives original values."""
        angle_orig = np.radians(75)
        axis_orig = Vec3(1, 2, 3).normalized()
        q = angle_axis_to_quaternion(angle_orig, axis_orig)
        angle_rec, axis_rec = quaternion_to_angle_axis(q)
        assert np.isclose(angle_rec, angle_orig, atol=1e-6)
        assert np.allclose(axis_rec.data[:3], axis_orig.data[:3], atol=1e-6)

    def test_zero_angle_gives_identity(self):
        """Angle 0 → identity quaternion regardless of axis."""
        q = angle_axis_to_quaternion(0.0, [1, 0, 0])
        assert are_same_rotation(q, quaternion_identity())

    def test_rotation_consistency_with_matrix(self):
        """angle_axis quaternion and Mat4x4.rotation matrix give identical rotation."""
        angle = np.radians(70)
        axis = [1, 1, 0]  # not normalized — both methods normalize it
        q = angle_axis_to_quaternion(angle, axis)
        mat_from_q = quaternion_to_rotation_matrix(q)
        mat_direct = Mat4x4.rotation(angle, axis)
        assert np.allclose(mat_from_q.data, mat_direct.data, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# SLERP
# ══════════════════════════════════════════════════════════════════════════════

class TestSlerp:

    def test_slerp_t0_gives_q0(self):
        """slerp(q0, q1, 0) = q0."""
        q0 = Quaternion.rotation_y(np.radians(30))
        q1 = Quaternion.rotation_y(np.radians(90))
        result = slerp(q0, q1, 0.0)
        assert are_same_rotation(result, q0)

    def test_slerp_t1_gives_q1(self):
        """slerp(q0, q1, 1) = q1."""
        q0 = Quaternion.rotation_y(np.radians(30))
        q1 = Quaternion.rotation_y(np.radians(90))
        result = slerp(q0, q1, 1.0)
        assert are_same_rotation(result, q1)

    def test_slerp_t05_midpoint(self):
        """slerp(q0, q1, 0.5) gives midpoint quaternion (angle = midpoint)."""
        phi0, phi1 = np.radians(20), np.radians(80)
        q0 = Quaternion.rotation_y(phi0)
        q1 = Quaternion.rotation_y(phi1)
        q_mid = slerp(q0, q1, 0.5)
        q_expected = Quaternion.rotation_y((phi0 + phi1) / 2)
        assert are_same_rotation(q_mid, q_expected)

    def test_slerp_result_is_unit_quaternion(self):
        """slerp always returns a unit quaternion."""
        q0 = euler_to_quaternion(*np.radians((10, 20, 30)), "XYZ")
        q1 = euler_to_quaternion(*np.radians((60, 45, 90)), "XYZ")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q = slerp(q0, q1, t)
            assert is_unit_quaternion(q, atol=1e-6), f"slerp(t={t}) is not a unit quaternion"

    def test_slerp_short_path(self):
        """slerp takes the short path between opposite quaternions."""
        q0 = Quaternion.rotation_y(np.radians(10))
        q1_neg = -Quaternion.rotation_y(np.radians(20))  # same quaternion, different sign
        q1_pos = Quaternion.rotation_y(np.radians(20))
        angle_neg = angle_between_rotations(slerp(q0, q1_neg, 0.5), q0)
        angle_pos = angle_between_rotations(slerp(q0, q1_pos, 0.5), q0)
        assert np.isclose(angle_neg, angle_pos, atol=1e-5)

    def test_slerp_nearly_identical_quaternions(self):
        """slerp is stable for nearly identical quaternions (dot ≈ 1)."""
        q0 = Quaternion.rotation_z(np.radians(45))
        q1 = Quaternion.rotation_z(np.radians(45.0001))
        result = slerp(q0, q1, 0.5)
        assert is_unit_quaternion(result, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# NLERP
# ══════════════════════════════════════════════════════════════════════════════

class TestNlerp:

    def test_nlerp_t0_gives_q0(self):
        """nlerp(q0, q1, 0) = q0."""
        q0 = Quaternion.rotation_x(np.radians(30))
        q1 = Quaternion.rotation_x(np.radians(90))
        assert are_same_rotation(nlerp(q0, q1, 0.0), q0)

    def test_nlerp_t1_gives_q1(self):
        """nlerp(q0, q1, 1) = q1."""
        q0 = Quaternion.rotation_x(np.radians(30))
        q1 = Quaternion.rotation_x(np.radians(90))
        assert are_same_rotation(nlerp(q0, q1, 1.0), q1)

    def test_nlerp_result_is_unit_quaternion(self):
        """nlerp always returns a unit quaternion."""
        q0 = euler_to_quaternion(*np.radians((10, 20, 30)), "ZYX")
        q1 = euler_to_quaternion(*np.radians((60, 45, 90)), "ZYX")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q = nlerp(q0, q1, t)
            assert is_unit_quaternion(q, atol=1e-6), f"nlerp(t={t}) is not a unit quaternion"

    def test_nlerp_short_path(self):
        """nlerp takes the short path between opposite quaternions."""
        q0 = Quaternion.rotation_y(np.radians(10))
        q1_neg = -Quaternion.rotation_y(np.radians(20))
        q1_pos = Quaternion.rotation_y(np.radians(20))
        angle_neg = angle_between_rotations(nlerp(q0, q1_neg, 0.5), q0)
        angle_pos = angle_between_rotations(nlerp(q0, q1_pos, 0.5), q0)
        assert np.isclose(angle_neg, angle_pos, atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_quaternion_dot_identical(self):
        """Dot product of unit quaternion with itself = 1."""
        q = euler_to_quaternion(*np.radians((30, 45, 60)), "XYZ")
        assert np.isclose(quaternion_dot(q, q), 1.0, atol=1e-6)

    def test_quaternion_dot_opposite(self):
        """Dot product of q and -q = -1."""
        q = Quaternion.rotation_x(np.radians(45))
        assert np.isclose(quaternion_dot(q, -q), -1.0, atol=1e-6)

    def test_are_same_rotation_identical(self):
        """Identical quaternions → True."""
        q = Quaternion.rotation_y(np.radians(60))
        assert are_same_rotation(q, q)

    def test_are_same_rotation_negative(self):
        """q and -q → True (double cover of SO(3))."""
        q = Quaternion.rotation_z(np.radians(45))
        assert are_same_rotation(q, -q)

    def test_are_same_rotation_different(self):
        """Different rotations → False."""
        q0 = Quaternion.rotation_x(np.radians(30))
        q1 = Quaternion.rotation_x(np.radians(90))
        assert not are_same_rotation(q0, q1)

    def test_angle_between_rotations_same(self):
        """Angle between identical rotations = 0."""
        q = Quaternion.rotation_z(np.radians(45))
        assert np.isclose(angle_between_rotations(q, q), 0.0, atol=1e-6)

    def test_angle_between_rotations_90deg(self):
        """Angle between Ry(0°) and Ry(90°) = 90°."""
        q0 = quaternion_identity()
        q1 = Quaternion.rotation_y(np.radians(90))
        angle = angle_between_rotations(q0, q1)
        assert np.isclose(np.degrees(angle), 90.0, atol=1e-4)

    def test_angle_between_rotations_opposite_sign(self):
        """q and -q have angle 0 between them (same rotation)."""
        q = Quaternion.rotation_x(np.radians(60))
        assert np.isclose(angle_between_rotations(q, -q), 0.0, atol=1e-6)

    def test_angle_between_rotations_symmetry(self):
        """angle_between_rotations is symmetric."""
        q0 = Quaternion.rotation_y(np.radians(30))
        q1 = Quaternion.rotation_z(np.radians(60))
        assert np.isclose(
            angle_between_rotations(q0, q1),
            angle_between_rotations(q1, q0),
            atol=1e-6
        )

    def test_quaternion_identity_is_unit(self):
        """quaternion_identity() is a unit quaternion."""
        assert is_unit_quaternion(quaternion_identity())

    def test_is_unit_quaternion_true(self):
        """Normalized quaternion is a unit quaternion."""
        q = Quaternion(1, 2, 3, 4).normalized()
        assert is_unit_quaternion(q)

    def test_is_unit_quaternion_false(self):
        """Non-normalized quaternion is not a unit quaternion."""
        q = Quaternion(1, 2, 3, 4)  # norm > 1
        assert not is_unit_quaternion(q)

    def test_quaternion_identity_no_rotation(self):
        """quaternion_identity() → zero rotation angle."""
        angle, _ = quaternion_to_angle_axis(quaternion_identity())
        assert np.isclose(angle, 0.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests (euler → quaternion → matrix consistency)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_euler_quat_matrix_consistency(self):
        """Rotation matrices obtained via quaternion and directly coincide."""
        angles = np.radians((30, 45, 60))
        for cfg in ALL_CONFIGS:
            mat_direct = Mat4x4.rotation_euler(*angles, configuration=cfg)
            q = euler_to_quaternion(*angles, cfg)
            mat_via_quat = quaternion_to_rotation_matrix(q)
            assert np.allclose(mat_direct.data, mat_via_quat.data, atol=1e-6), \
                f"{cfg}: matrices via quaternion and directly differ"

    def test_rotate_vector_quaternion_vs_matrix(self):
        """Vector rotation via quaternion and matrix gives identical result."""
        phi, theta, psi = np.radians((30, 45, 60))
        q = euler_xyz_to_quaternion(phi, theta, psi)
        mat = Mat4x4.rotation_euler(phi, theta, psi, "XYZ")

        from src.math.Vec3 import Vec3
        v = Vec3(1, 2, 3)
        v_quat = q.rotate_vector(v)
        v_mat = mat @ v
        assert np.allclose(v_quat.data[:3], v_mat.data[:3], atol=1e-6)

    def test_composition_euler_via_quaternions(self):
        """Sequential application of two rotations via quaternions = product matrix."""
        phi, theta = np.radians(45), np.radians(30)
        q1 = Quaternion.rotation_x(phi)
        q2 = Quaternion.rotation_y(theta)
        q_composed = q1 * q2

        mat1 = Mat4x4.rotation_x(phi)
        mat2 = Mat4x4.rotation_y(theta)
        mat_composed = mat1 * mat2

        mat_from_q = quaternion_to_rotation_matrix(q_composed)
        assert np.allclose(mat_from_q.data, mat_composed.data, atol=1e-6)

    def test_slerp_endpoints_match_euler(self):
        """SLERP endpoints match quaternions built from Euler angles."""
        q0 = euler_to_quaternion(*np.radians((10, 20, 30)), "ZYX")
        q1 = euler_to_quaternion(*np.radians((60, 45, 90)), "ZYX")
        assert are_same_rotation(slerp(q0, q1, 0.0), q0)
        assert are_same_rotation(slerp(q0, q1, 1.0), q1)
