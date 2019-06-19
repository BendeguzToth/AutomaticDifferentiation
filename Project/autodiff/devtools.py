"""
For development purposes.
"""

# Standard libraries
from functools import wraps
import logging

# Project files
import autodiff.constants as const

LOGGER = logging.getLogger("autodiff_dev")


def devmodeonly(func):
    """
    Apply this decorator to your decorator to turn it
    off when dev mode is False.
    :param func:
    :return:
    """
    if const.devmode:
        return func
    else:
        return lambda f: f


@devmodeonly
def logging(func):
    """
    Decorator.
    Logs every function call.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        LOGGER.debug(f"'{func.__module__}.{func.__name__}' called with arguments: {args} and kw arguments: {kwargs}")
        return func(*args, **kwargs)
    return inner


@devmodeonly
def unstable(func):
    """
    Decorator.
    Logs warning on first call.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        if not inner.called:
            inner.called = True
            LOGGER.warning(f"Unstable function '{func.__module__}.{func.__name__}' called.")
        return func(*args, **kwargs)
    inner.called = False
    return inner


def placeholder(func):
    """
    Decorator.
    Use to decorate unimplemented placeholder
    functions. Will thor NotImplementedError on call.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        raise NotImplementedError(f"'{func.__module__}.{func.__name__}' is not yet implemented.")
    return inner
