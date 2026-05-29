import numpy as np
import jax.numpy as jnp


class Cuboid:
    """
    Axis-aligned cuboid geometry with corners at (0,0,0) and (Lx,Ly,Lz).

    Provides the same interface as cube.py (gb, box_geom, bounds, dSlabs)
    but parameterised by side lengths.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Side lengths along x, y, z.  Default to 1.0 (unit cube).
    """

    def __init__(self, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0):
        self.bnds = [[0.0, 0.0, 0.0], [float(Lx), float(Ly), float(Lz)]]

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def gb_np(self, p: np.ndarray) -> np.ndarray:
        """Return boolean mask: True where point rows lie on the boundary (NumPy)."""
        b = self.bnds
        return (
            (np.abs(p[:, 0] - b[0][0]) < 1e-14) | (np.abs(p[:, 0] - b[1][0]) < 1e-14) |
            (np.abs(p[:, 1] - b[0][1]) < 1e-14) | (np.abs(p[:, 1] - b[1][1]) < 1e-14) |
            (np.abs(p[:, 2] - b[0][2]) < 1e-14) | (np.abs(p[:, 2] - b[1][2]) < 1e-14)
        )

    def gb_jnp(self, p):
        """Return boolean mask: True where point rows lie on the boundary (JAX)."""
        b = self.bnds
        return (
            (jnp.abs(p[..., 0] - b[0][0]) < 1e-14) | (jnp.abs(p[..., 0] - b[1][0]) < 1e-14) |
            (jnp.abs(p[..., 1] - b[0][1]) < 1e-14) | (jnp.abs(p[..., 1] - b[1][1]) < 1e-14) |
            (jnp.abs(p[..., 2] - b[0][2]) < 1e-14) | (jnp.abs(p[..., 2] - b[1][2]) < 1e-14)
        )

    def gb(self, p, jax_avail: bool = True, torch_avail: bool = False):
        """Dispatch boundary detection to JAX or NumPy backend."""
        if jax_avail:
            return self.gb_jnp(p)
        else:
            return self.gb_np(p)

    # ------------------------------------------------------------------
    # Geometry accessors
    # ------------------------------------------------------------------

    def box_geom(self, jax_avail: bool = True, torch_avail: bool = True):
        """Return the bounding-box array [[0,0,0],[Lx,Ly,Lz]]."""
        if jax_avail:
            return jnp.array(self.bnds)
        else:
            return np.array(self.bnds)

    def bounds(self):
        """Return the raw bounding-box list [[0,0,0],[Lx,Ly,Lz]]."""
        return self.bnds

    # ------------------------------------------------------------------
    # Domain decomposition
    # ------------------------------------------------------------------

    def dSlabs(self, N: int):
        """
        Decompose the x-extent into N slabs.

        Returns
        -------
        slabs : list of [[xlo,ylo,zlo],[xhi,yhi,zhi]]
            Bounding boxes of the N-1 interior slabs.
        connectivity : list of [left_neighbour, right_neighbour]
            -1 indicates a domain boundary (no neighbour).
        H : float
            Slab width in x.
        """
        b = self.bnds
        H = (b[1][0] - b[0][0]) / N
        slabs = []
        connectivity = []
        for n in range(N - 1):
            c = b[0][0] + (n + 1) * H
            slab = [[c - H, b[0][1], b[0][2]], [c + H, b[1][1], b[1][2]]]
            if n == 0:
                connectivity.append([-1, 1])
            elif n == N - 2:
                connectivity.append([n - 1, -1])
            else:
                connectivity.append([n - 1, n + 1])
            slabs.append(slab)
        return slabs, connectivity, H