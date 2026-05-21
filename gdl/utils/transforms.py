"""
Point cloud augmentation transforms.

All transforms operate on numpy arrays of shape (N, 3) and return (N, 3).
They can be composed with ComposeTransforms.
"""

import numpy as np


class RandomRotation:
    """
    Randomly rotate the point cloud.

    Args:
        mode : 'so3'   — full random rotation from SO(3)
               'z'     — rotation only around vertical (Z) axis  [default]
               'perturb' — small random perturbation (±max_angle degrees)
        max_angle : Used when mode='perturb', in degrees
    """

    def __init__(self, mode: str = "z", max_angle: float = 15.0):
        assert mode in ("so3", "z", "perturb"), f"Unknown rotation mode: {mode}"
        self.mode = mode
        self.max_angle = max_angle

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if self.mode == "so3":
            R = self._random_so3()
        elif self.mode == "z":
            R = self._random_z()
        else:
            R = self._random_perturb()
        return points @ R.T

    @staticmethod
    def _random_so3() -> np.ndarray:
        """Uniform random rotation from SO(3) via QR decomposition."""
        M = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(M)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q.astype(np.float32)

    @staticmethod
    def _random_z() -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    def _random_perturb(self) -> np.ndarray:
        max_rad = np.deg2rad(self.max_angle)
        angles = np.random.uniform(-max_rad, max_rad, size=3)
        Rx = self._rot_x(angles[0])
        Ry = self._rot_y(angles[1])
        Rz = self._rot_z(angles[2])
        return (Rz @ Ry @ Rx).astype(np.float32)

    @staticmethod
    def _rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def _rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class RandomJitter:
    """
    Add independent Gaussian noise to each point.

    Args:
        sigma : Standard deviation of the noise
        clip  : Clip noise to [-clip, clip] to avoid extreme outliers
    """

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points: np.ndarray) -> np.ndarray:
        noise = np.clip(
            np.random.randn(*points.shape) * self.sigma,
            -self.clip,
            self.clip,
        )
        return (points + noise).astype(np.float32)


class RandomScale:
    """
    Uniformly scale the point cloud by a random factor in [lo, hi].
    """

    def __init__(self, lo: float = 0.8, hi: float = 1.2):
        self.lo = lo
        self.hi = hi

    def __call__(self, points: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.lo, self.hi)
        return (points * scale).astype(np.float32)


class ComposeTransforms:
    """Apply a list of transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            points = t(points)
        return points


def get_train_transform(
    rotation_mode: str = "z",
    jitter_sigma: float = 0.01,
    scale_lo: float = 0.8,
    scale_hi: float = 1.2,
) -> ComposeTransforms:
    """Standard augmentation pipeline for training."""
    return ComposeTransforms([
        RandomScale(lo=scale_lo, hi=scale_hi),
        RandomRotation(mode=rotation_mode),
        RandomJitter(sigma=jitter_sigma),
    ])
