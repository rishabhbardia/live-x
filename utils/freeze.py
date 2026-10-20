from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

def check_shape(
        array: np.ndarray,
        rows: int | None = None,
        cols: int | None = None,
    ) -> None:
        """Validate shape and finite values. rows=None allows any length."""
        expected = (rows,) if cols is None else (rows, cols)

        if array.ndim != len(expected) or any(
            size is not None and actual != size
            for actual, size in zip(array.shape, expected)
        ):
            raise ValueError(
                f"Expected shape {expected} (None means any size), "
                f"got {array.shape}"
            )

        if not np.isfinite(array).all():
            raise ValueError("Array contains NaN or infinite values")

def freeze_array(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected ndarray, got {type(array).__name__}")

    if array.dtype.hasobject:
        raise TypeError("Object arrays need a schema-specific conversion")

    array.setflags(write=False)
    return array


def freeze_data(value: Any) -> Any:
    """Read-only conversion for plain containers and numeric NumPy data."""
    if isinstance(value, np.ndarray):
        return freeze_array(value)

    if isinstance(value, Mapping):
        return MappingProxyType({
            key: freeze_data(item)
            for key, item in value.items()
        })

    if isinstance(value, (list, tuple)):
        return tuple(freeze_data(item) for item in value)

    if value is None or isinstance(
        value, (str, bytes, bool, int, float, complex, np.number, np.bool_)
    ):
        return value

    raise TypeError(
        f"No read-only conversion defined for {type(value).__name__}"
    )

def frozen_array(a: Any, dtype: Any = np.float64) -> np.ndarray:
    """A read-only COPY, for small values the caller may still be holding."""
    out = np.array(a, dtype=dtype)
    out.setflags(write=False)
    return out