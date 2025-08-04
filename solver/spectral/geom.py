from abc import ABCMeta, abstractmethod, abstractproperty

############################################################################################
# Abstract base class for geometries: defines minimal interface (bounds property)
############################################################################################
class AbstractGeometry(metaclass=ABCMeta):
    @abstractproperty
    def bounds(self):
        """
        Returns:
            A 2×ndim array specifying the lower and upper bounds of the domain.
        """
        pass


############################################################################################
# BoxGeometry: simple axis-aligned box defined entirely by its bounds
############################################################################################
class BoxGeometry(AbstractGeometry):
    def __init__(self, box_geom):
        """
        Parameters:
            box_geom: numpy.ndarray of shape (2, ndim), where
                      box_geom[0] = lower-left corner in each dimension,
                      box_geom[1] = upper-right corner in each dimension.
        """
        self.box_geom = box_geom

    @property
    def bounds(self):
        """
        Return the stored box bounds (2×ndim array).
        """
        return self.box_geom

