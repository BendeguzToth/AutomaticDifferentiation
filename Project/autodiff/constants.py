"""
This file defines some constants that
are used across the selfdiff project.
"""

# Logic
dtype = "float64"   # Default data type used by Tensors.
fuzz = 1e-7         # Small number added to values to prevent
                    # division by zero, or zero in log.

# Python
devmode = False     # Developer mode, enables logging and warnings.
