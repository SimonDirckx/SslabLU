class slabGeometry:

    """
    
    A class used to represent the geometry of a double slab
    for now: no restriction on map
    
    """

    def __init__(self,l2g,bounds):
         self._l2g      = l2g
         self.bounds    = bounds

    def l2g(self,x,y):
        return self._l2g(x,y)