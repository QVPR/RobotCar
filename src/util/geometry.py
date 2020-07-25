import numpy as np
from scipy.spatial.transform import Rotation


class SE3:
    def __init__(self, t, R):
        self._single = False

        if t.ndim not in [1, 2] or t.shape[-1] != 3:
            raise ValueError("Expected `t` to have shape (3,) or (N x 3), "
                             "got {}.".format(t.shape))

        # If a single translation is given, convert it to a 2D 1 x 3 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods
        if t.shape == (3,):
            t = t[None, :]
            self._single = True
            if len(R) > 1:            
                raise ValueError("Different number of translations 1 and rotations {}.".format(len(R)))
        elif len(t) == 1:
            self._single = True
        else:
            if len(t) != len(R):
                raise ValueError("Differing number of translations {} and rotations {}".format(len(t),len(R)))
        self._t = t
        self._R = R
        self.len = len(R)

    @classmethod
    def from_xyzrpy(cls, xyzrpy):
        t = xyzrpy[:, :3] 
        R = Rotation.from_euler('ZYX', xyzrpy[:, 3:])
        return cls(t, R)

    @classmethod
    def from_xyzquat(cls, t, quat):
        R = Rotation.from_quat(quat) 
        return cls(t, R)

    @classmethod
    def from_mat(cls, T):
        R = Rotation.from_quat(T[:, :3, :3])    
        t = T[:, :3, 3]
        return cls(t, R)

    def __getitem__(self, indexer):
        return self.__class__(self.t()[indexer], self.R()[indexer])

    def __len__(self):
        return self.len

    def __mul__(self, other):
        """
        Performs element-wise pose composition.
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} transformations in "
                             "second object.".format(
                                len(self), len(other)))
        return self.__class__(self.R().apply(other.t()) + self.t(), self.R() * other.R())

    def __truediv__(self, other):
        """
        Computes relative pose, similar to MATLAB convention (x = A \ b for Ax = b). Example:
        T1 / T2 = T1.inv() * T2
        TO DO: Broadcasting
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} transformations in "
                             "second object.".format(
                                len(self), len(other)))
        R1_inv = self.R().inv()
        t_new = R1_inv.apply(other.t() - self.t())
        return self.__class__(t_new, R1_inv * other.R())

    def t(self):
        return self._t[0] if self._single else self._t

    def R(self):
        return self._R

    def inv(self):
        R_inv = self.R().inv()
        t_new = -R_inv.apply(self.t())
        return SE3(t_new, R_inv)

    def components(self):
        return self.t(), self.R()

    def magnitude(self):
        return np.linalg.norm(self.t(), axis=-1), self.R().magnitude()


def metric(p1, p2, w):
    """
    Computes metric on the cartesian product space representation of SE(3).
    Args:
        p1 (SE3) : set of poses
        p2 (SE3) : set of poses (same size as p1)
        w (float > 0) : weight for attitude component
    """
    if not(len(p1) == 1 or len(p2) == 1 or len(p1) == len(p2)):
        raise ValueError("Expected equal number of transformations in both "
                            "or a single transformation in either object, "
                            "got {} transformations in first and {} transformations in "
                            "second object.".format(
                            len(p1), len(p2)))
    if w < 0:
        raise ValueError("Weight must be non-negative, currently {}".format(w))
    p_rel = p1 / p2
    t_dist = np.linalg.norm(p_rel.t(), axis=-1)
    R_dist = p_rel.R().magnitude()
    return t_dist + w * R_dist 


def error(p1, p2):
    if not(len(p1) == 1 or len(p2) == 1 or len(p1) == len(p2)):
        raise ValueError("Expected equal number of transformations in both "
                            "or a single transformation in either object, "
                            "got {} transformations in first and {} transformations in "
                            "second object.".format(
                            len(p1), len(p2)))
    p_rel = p1 / p2
    return p_rel.magnitude()


def combine(listOfPoses):
    tList = []
    qList = []
    for pose in listOfPoses:
        tList.append(pose.t())
        qList.append(pose.R().as_quat())
    return SE3(np.asarray(tList), Rotation.from_quat(np.asarray(qList)))
