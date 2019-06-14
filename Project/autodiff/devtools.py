"""
For development purposes.
"""

# Standard libraries
from functools import wraps
import logging

# Macros
DEV_MODE = True
LOGGER = logging.getLogger("autodiff_dev")
ENABLE_DECORATORS = True


def undecorate(func):
    """
    Disables the decorator when ENABLE_DECORATORS is False.
    """
    if ENABLE_DECORATORS:
        return func
    else:
        return lambda f: f


@undecorate
def log(func):
    """
    Decorator.
    Logs every function call.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        LOGGER.debug(f"'{func.__module__}.{func.__name__}' called with arguments: {args} and kw arguments: {kwargs}")
        return func(*args, **kwargs)
    return inner


@undecorate
def unstable(func):
    """
    Decorator.
    If DEV_MODE is False, raises exception
    on calling the function. Logs warning
    on first call otherwise.
    """
    if DEV_MODE:
        @wraps(func)
        def inner(*args, **kwargs):
            if not inner.called:
                inner.called = True
                LOGGER.warning(f"Unstable function '{func.__module__}.{func.__name__}' called.")
            return func(*args, **kwargs)
        inner.called = False
        return inner
    else:
        @wraps(func)
        def inner(*args, **kwargs):
            raise NotImplementedError(f"'{func.__module__}.{func.__name__}' is under development.")
        return inner


@undecorate
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
