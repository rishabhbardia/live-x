import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np
from numpy.testing import assert_allclose

from data.types import CameraIntrinsics, CameraPose
from geometry.projections import EPS, plane_ray_depth
from geometry.pose import plane_warp, omega_of

from scipy.spatial.transform import Rotation as R

def assert_close(actual, expected):
    assert_allclose(actual, expected, rtol=0, atol=1e-10)


def make_intrinsics():
    return CameraIntrinsics(
        f=320.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )


def make_pose(intr, omega, C, covariance=None):
    return CameraPose(
        index=0,
        omega=omega,
        C=C,
        intr=intr,
        reproj_px=0.0,
        samples=4,
        covariance=covariance,
    )


class TestPlaneRay(unittest.TestCase):

    def setUp(self):
        self.intr = make_intrinsics()

    def test_warp(self):

        C = np.asarray([0,0,0.4])
        R = np.eye(3,3)

        O = np.asarray([0,0,3])
        e1 = np.asarray([0,-1.0, 0.0])
        e2 = np.asarray([1.0, 0.0, 0.0])
        
        d = plane_ray_depth(uv=np.asarray([160, 120]).reshape(-1,2),
                        R=R, C=C, f=self.intr.f, 
                        cx=self.intr.cx,cy=self.intr.cy, 
                        O=O, n_world=np.cross(e1,e2))
        assert_close(d, 2.6)
        return

class TestPlaneWarp(unittest.TestCase):

    def setUp(self):
        self.intr = make_intrinsics()

    def test_warp(self):

        C_ref = np.asarray([0,0,0.4])
        C = np.asarray([1,2,4])
        omega_ref = omega_of(np.eye(3,3))
        omega = omega_of(R.from_euler("XYZ", [10, 0, 0], degrees=True).as_matrix() @ np.eye(3,3))
        O = np.asarray([0,0,3])
        e1 = np.asarray([0,-1.0, 0.0])
        e2 = np.asarray([1.0, 0.0, 0.0])
        
        H = plane_warp(
            K=self.intr.K,
            omega_ref=omega_ref,
            C_ref=C_ref,
            omega=omega,
            C=C,
            basis=(O,e1,e2)
        )

        # TODO : Add a point on the plane and see if the 2D projections of that point
        #        are descriped by H

        return

class TestCameraIntrinsics(unittest.TestCase):
    def setUp(self):
        self.intr = make_intrinsics()

    def test_matrix_size_and_fov(self):
        assert_close(
            self.intr.K,
            [
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
        )
        self.assertEqual(self.intr.K.dtype, np.dtype(np.float64))
        self.assertEqual(self.intr.size, (640, 480))
        self.assertAlmostEqual(self.intr.hfov_deg, 90.0)

    def test_read_only(self):
        with self.assertRaises(FrozenInstanceError):
            self.intr.f = 100.0

        with self.assertRaises(ValueError):
            self.intr.K[0, 0] = 100.0

        editable = self.intr.K.copy()
        editable[0, 0] = 100.0
        self.assertEqual(self.intr.K[0, 0], 320.0)

    def test_replace_rebuilds_matrix(self):
        updated = replace(self.intr, f=640.0)

        assert_close(
            updated.K,
            [
                [640.0, 0.0, 320.0],
                [0.0, 640.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
        )
        self.assertEqual(self.intr.f, 320.0)
        self.assertEqual(self.intr.K[0, 0], 320.0)
        self.assertFalse(updated.K.flags.writeable)

    def test_invalid_intrinsics(self):
        for f in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(f=f):
                with self.assertRaises(ValueError):
                    replace(self.intr, f=f)

        for changes in (
            {"cx": np.nan},
            {"cy": np.inf},
            {"width": 0},
            {"height": -1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(self.intr, **changes)


class TestCameraPose(unittest.TestCase):
    def setUp(self):
        self.intr = make_intrinsics()
        self.C = np.array([10.0, 20.0, 30.0])

        # +90 degrees about Y, world -> camera.
        self.pose = make_pose(
            self.intr,
            omega=np.array([0.0, np.pi / 2.0, 0.0]),
            C=self.C,
        )

        # Independent expected rotation, not obtained from pose.R.
        self.expected_R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])

        # Corresponding camera coordinates:
        # (0, 0, 2), (1, -0.5, 5), (-2, 1, 4), (10, 0, 2).
        self.world = np.array([
            [8.0, 20.0, 30.0],
            [5.0, 19.5, 31.0],
            [6.0, 21.0, 28.0],
            [8.0, 20.0, 40.0],
        ])

        self.pixels = np.array([
            [320.0, 240.0],
            [384.0, 208.0],
            [160.0, 320.0],
            [1920.0, 240.0],  # Outside image, but still projectable.
        ])

        self.depth = np.array([2.0, 5.0, 4.0, 2.0])

    def test_rotation_and_forward(self):
        assert_close(self.pose.R, self.expected_R)
        assert_close(self.pose.forward, [-1.0, 0.0, 0.0])
        assert_close(self.pose.R @ self.pose.R.T, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(self.pose.R), 1.0)
        self.assertIsNone(self.pose.covariance)

    def test_project_known_points(self):
        pixels, depth = self.pose.project(self.world)

        self.assertEqual(pixels.shape, (4, 2))
        self.assertEqual(depth.shape, (4,))
        assert_close(pixels, self.pixels)
        assert_close(depth, self.depth)

    def test_unproject_known_pixels(self):
        world = self.pose.unproject(self.pixels, self.depth)

        self.assertEqual(world.shape, (4, 3))
        assert_close(world, self.world)

    def test_project_unproject_round_trip(self):
        rng = np.random.default_rng(42)

        camera_points = rng.uniform(-2.0, 2.0, size=(32, 3))
        camera_points[:, 2] = rng.uniform(1.0, 10.0, size=32)
        world = camera_points @ self.expected_R + self.C

        pixels, depth = self.pose.project(world)
        reconstructed = self.pose.unproject(pixels, depth)

        assert_close(reconstructed, world)

    def test_rays(self):
        directions = self.pose.rays(self.pixels)

        expected = self.world - self.C
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)

        self.assertEqual(directions.shape, (4, 3))
        assert_close(directions, expected)
        assert_close(np.linalg.norm(directions, axis=1), np.ones(4))

        # The principal-point ray is the camera's forward direction.
        assert_close(directions[0], self.pose.forward)

        # Unit-ray distance is NOT generally equal to optical Z-depth.
        camera_directions = directions @ self.expected_R.T
        distances = self.depth / camera_directions[:, 2]

        reconstructed = self.C + distances[:, None] * directions
        assert_close(reconstructed, self.world)

        # The second pixel is off-axis, so range must exceed Z-depth.
        self.assertGreater(distances[1], self.depth[1])

    def test_invalid_projection_depths(self):
        pose = make_pose(
            self.intr,
            omega=np.zeros(3),
            C=np.zeros(3),
        )

        points = np.zeros((5, 3))
        points[:, 2] = [0.0, -1.0, EPS / 2.0, EPS, 2.0 * EPS]

        pixels, depth = pose.project(points)

        # Contract: z <= EPS gives NaN pixels, but preserves returned Z.
        self.assertTrue(np.isnan(pixels[:4]).all())
        assert_close(pixels[4], [320.0, 240.0])
        assert_close(depth, points[:, 2])

    def test_empty_batches(self):
        pixels, depth = self.pose.project(np.empty((0, 3)))

        self.assertEqual(pixels.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))
        self.assertEqual(
            self.pose.unproject(pixels, depth).shape, (0, 3)
        )
        self.assertEqual(self.pose.rays(pixels).shape, (0, 3))

    def test_invalid_inputs(self):
        for points in (
            np.zeros(3),
            np.zeros((2, 2)),
            np.full((1, 3), np.nan),
        ):
            with self.subTest(operation="project", shape=points.shape):
                with self.assertRaises(ValueError):
                    self.pose.project(points)

        for pixels in (
            np.zeros(2),
            np.zeros((1, 3)),
            np.full((1, 2), np.nan),
        ):
            with self.subTest(operation="rays", shape=pixels.shape):
                with self.assertRaises(ValueError):
                    self.pose.rays(pixels)

            with self.subTest(operation="unproject", shape=pixels.shape):
                with self.assertRaises(ValueError):
                    self.pose.unproject(pixels, np.ones(1))

        pixels = np.zeros((2, 2))

        for depth in (
            np.array(1.0),          # Scalar.
            np.ones((2, 1)),        # Wrong rank.
            np.ones(3),             # Wrong point count.
            np.array([1.0, np.inf]),
        ):
            with self.subTest(operation="depth", shape=depth.shape):
                with self.assertRaises(ValueError):
                    self.pose.unproject(pixels, depth)

    def test_read_only(self):
        with self.assertRaises(FrozenInstanceError):
            self.pose.C = np.zeros(3)

        for name in ("omega", "C", "R", "forward"):
            with self.subTest(field=name):
                array = getattr(self.pose, name)
                self.assertFalse(array.flags.writeable)

                with self.assertRaises(ValueError):
                    array.flat[0] = 99.0

    def test_owns_input_arrays_and_covariance(self):
        # Pass views to catch changes through a separate writable base.
        storage = np.zeros((2, 3))
        storage[1] = [1.0, 2.0, 3.0]
        covariance = np.eye(6)

        pose = make_pose(
            self.intr,
            omega=storage[0],
            C=storage[1],
            covariance=covariance,
        )

        storage[:] = 99.0
        covariance[:] = 99.0

        assert_close(pose.omega, np.zeros(3))
        assert_close(pose.C, [1.0, 2.0, 3.0])
        assert_close(pose.R, np.eye(3))
        assert_close(pose.covariance, np.eye(6))

        with self.assertRaises(ValueError):
            pose.covariance[0, 0] = 99.0


if __name__ == "__main__":
    unittest.main(verbosity=2)