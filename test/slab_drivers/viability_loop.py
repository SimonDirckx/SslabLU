import os

def run_driver(ntiles_unit, p=10, ndim=3, b=20):
    """
    Run timing_viability_driver.py with calculated H value.

    Parameters:
        ntiles_unit (int): number of tiles per unit
        p (int): parameter p (default=10)
        ndim (int): number of dimensions (default=3)
        b (float): value for numerator in H calculation (default=20)
    """
    H = b / (ntiles_unit * p)
    os.system(
        f"python timing_viability_driver.py --H {H:.10f} --ntiles_unit {ntiles_unit} --p {p} --ndim {ndim}"
    )

# Example usage:
run_driver(20, b=20)
run_driver(20, b=40)
run_driver(30, b=20)
run_driver(30, b=40)
run_driver(40, b=20)
run_driver(40, b=40)
