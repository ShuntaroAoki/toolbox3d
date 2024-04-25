"""3D Metrics."""


import numpy as np


class ChamferDistance(object):
    """3D Chamfer distance."""
    def init(self) -> None:
        pass

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.calc(a, b)
    
    def calc(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate 3D Chamfer distance.
        
        Parameters
        ----------
        a, b: np.ndarray
            Point clouds, points x dim (xyz).

        Returns
        -------
        float
        """
        _a = np.expand_dims(a, axis=1)
        _b = np.expand_dims(b, axis=0)
        dist_matrix = np.sum(np.square(_a - _b), 2)
        dist_ab = np.min(dist_matrix, axis=1)
        dist_ba = np.min(dist_matrix, axis=0)
        return np.mean(dist_ab) + np.mean(dist_ba)


def chamfer_distance(a, b) -> float:
    """Chamfer distance between a and b."""
    return ChamferDistance()(a, b)
