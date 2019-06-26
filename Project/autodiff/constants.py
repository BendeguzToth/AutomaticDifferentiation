"""
This file defines some constants that
are used across the selfdiff project.
"""

# Third-party dependencies
import numpy as np

# Logic
dtype = "float64"   # Default data type used by Tensors.
fuzz = 1e-7         # Small number added to values to prevent
                    # division by zero, or zero in log.
large = 1e9         # Used to mimic +/- infinity. For example masking
                    # inside softmax.
avg = np.mean       # Method to average in backward pass.
sum = np.sum        # Method to sum in backward pass.

# Python
devmode = True     # Developer mode, enables logging and warnings.
