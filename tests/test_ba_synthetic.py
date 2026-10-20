import unittest

import cv2
import numpy as np

from data.types import AnnotationPriors, CameraLock, SolverConfig, SolverParameters
from data.venue import VenueGeometryStructure, VenueMeasurements
from geometry.pose import look_at
from geometry.solver import staged_solve

def _project(points, R, C, intr):
    """Independent pinhole calculation: deliberately do not call the costs."""
    camera = (points - C) @ R.T
    assert np.all(camera[:, 2] > 0), "Synthetic points must be in front of camera"
    f, cx, cy = intr
    return f * camera[:, :2] / camera[:, 2, None] + [cx, cy]


class testStagedSolve(unittest.TestCase):

    def test_noiseless(self):
        self._check_recovery(noise_px=0.0)

    def test_with_pixel_noise(self):
        self._check_recovery(noise_px=0.5)

    def _check_recovery(self, noise_px):
        rng = np.random.default_rng(7)
        image_wh = (1920, 1080)
        intr_true = np.array([1300.0, 960.0, 540.0])
        C_true = np.array([-5.3, -6.8, 2.4])
        theta_true = np.deg2rad([0.0, 31.0, 73.0, 112.0])
        Q_true = np.array([
            [-2.0, -1.5], [0.0, -1.5], [2.0, -1.0],
            [-1.7, 1.0], [0.3, 1.8], [2.1, 1.3],
        ])

        rotations = np.array([
            look_at(C_true, target)
            for target in [(-0.8, -0.4, 0.5), (0.0, 0.6, 0.8), (0.9, -0.2, 0.3)]
        ])

        n_frames, n_pillars, n_tracks = len(rotations), len(theta_true), len(Q_true)

        measurements = VenueMeasurements(
            diameter=9.1, pillar_height=1.83, shape="circle", total_posts=8,
        )

        priors = AnnotationPriors(
            pillar_base_sigma_px=1.0,
            pillar_top_sigma_px=1.0,
            landmark_sigma_px=1.0,
            principal_point_sigma_px=75.0,
            delta_R_m=0.25,
            delta_H_m=0.05,
            pillar_ids=tuple(range(n_pillars)),
        )
        structure = VenueGeometryStructure(measurements, priors)

        base_true = np.column_stack([
            measurements.radius * np.cos(theta_true),
            measurements.radius * np.sin(theta_true),
            np.zeros(n_pillars),
        ])
        top_true = base_true + [0.0, 0.0, measurements.pillar_height]

        canvas_true = np.column_stack([Q_true, np.zeros(n_tracks)])

        observations = {
            "frames": [f"frame_{i:05d}" for i in range(n_frames)],
            "pillar_ids": list(range(n_pillars)),
            "track_ids": list(range(n_tracks)),
            "pil_idx": np.array([(i, j) for i in range(n_frames)
                                 for j in range(n_pillars)], dtype=np.int64),
            "pt_idx": np.array([(i, j) for i in range(n_frames)
                                for j in range(n_tracks)], dtype=np.int64),
        }

        clean_uv = {}
        for key, points in [("pil_base", base_true), ("pil_top", top_true),
                            ("pt_uv", canvas_true)]:
            pixels = np.concatenate([
                _project(points, R, C_true, intr_true) for R in rotations
            ])
            self.assertTrue(np.all((pixels >= 0) & (pixels < image_wh)))
            clean_uv[key] = pixels
            observations[key] = pixels + rng.normal(0.0, noise_px, pixels.shape)

        scene, problem, keep, summary = staged_solve(
            params=SolverParameters(),
            focal_grid=np.array([1000.0]),
            observations=observations,
            structure=structure,
            venue_measurements=measurements,
            annotation_priors=priors,
            image_wh=image_wh,
            use_delta=False,
            use_dh=False,
            verbose=False,
            lock=CameraLock.ROT,
        )

        self.assertTrue(summary.IsSolutionUsable(), summary.BriefReport())
        self.assertEqual(scene.lock, CameraLock.ROT)
        stages = (SolverConfig.PILLARS_ONLY, SolverConfig.PILLARS_LANDMARKS,
                  SolverConfig.JOINT)
        self.assertEqual(set(scene.cost_history), set(stages))
        self.assertTrue(np.isfinite(list(scene.cost_history.values())).all())

        self.assertLessEqual(scene.cost_history[stages[2]],
                             scene.cost_history[stages[1]] + 1e-8)

        base_fit = np.column_stack([
            measurements.radius * np.cos(scene.thetas()),
            measurements.radius * np.sin(scene.thetas()),
            np.zeros(n_pillars),
        ])

        top_fit = base_fit + [0.0, 0.0, measurements.pillar_height]
        canvas_fit = np.column_stack([scene.tracked_points, np.zeros(n_tracks)])
        rotations_fit = [cv2.Rodrigues(omega)[0] for omega in scene.omega]

        errors = []
        for key, points in [("pil_base", base_fit), ("pil_top", top_fit),
                            ("pt_uv", canvas_fit)]:
            pixels = np.concatenate([
                _project(points, R, scene.C, scene.intr_of(i))
                for i, R in enumerate(rotations_fit)
            ])
            # Compare with clean ground truth, not just the noisy fit targets.
            errors.append(pixels - clean_uv[key])
        rms_px = np.sqrt(np.mean(np.sum(np.concatenate(errors) ** 2, axis=1)))

        rotation_error_deg = max(
            np.rad2deg(np.linalg.norm(cv2.Rodrigues(R @ R_true.T)[0]))
            for R, R_true in zip(rotations_fit, rotations)
        )

        center_error_m = np.linalg.norm(scene.C - C_true)
        landmark_error_m = np.max(np.linalg.norm(canvas_fit - canvas_true, axis=1))
        pillar_error_m = np.max(np.linalg.norm(base_fit - base_true, axis=1))
        focal_relative_error = abs(scene.f / intr_true[0] - 1.0)
        pp_error_px = np.linalg.norm(scene.principal_point - intr_true[1:])

        print(f"\nnoise={noise_px:g}px: clean reprojection RMS={rms_px:.5f}px, "
              f"focal error={100 * focal_relative_error:.4f}%, "
              f"center error={center_error_m:.5f}m, "
              f"rotation error={rotation_error_deg:.5f}deg, "
              f"landmark error={landmark_error_m:.5f}m")

        exact = noise_px == 0.0
        self.assertLess(rms_px, 1e-3 if exact else 1.0)
        self.assertLess(focal_relative_error, 1e-4 if exact else 0.03)
        self.assertLess(pp_error_px, 0.05 if exact else 15.0)
        self.assertLess(center_error_m, 1e-4 if exact else 1e-1)
        self.assertLess(rotation_error_deg, 1e-4 if exact else 0.5)

        return


if __name__ == "__main__":
    unittest.main(verbosity=2)