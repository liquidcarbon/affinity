__doc__ = """
Module for creating well-documented datasets, with types and annotations.
"""

import numpy as np
import pandas as pd
from typing import Union


class Descriptor:
    def __get__(self, instance, owner):
        return self if not instance else instance.__dict__[self.name]
    def __set__(self, instance, values):
        _values = values or []
        try:
            self._values = self.array_class(_values, dtype=self.dtype)
        except OverflowError as e:
            raise e
        except TypeError:
            self._values = self.array_class(_values)
        finally:
            self.size = len(self._values)
            if instance is not None:
                instance.__dict__[self.name] = self._values
    def __set_name__(self, owner, name):
        self.name = name


class Scalar(Descriptor):
    def __init__(self, dtype, value=None, comment=None, array_class=np.array):
        self.dtype = dtype
        self.value = value
        self.comment = comment
        self.array_class = array_class


class Vector(Descriptor):
    @classmethod
    def from_scalar(cls, scalar: Scalar, length=1):
        _value = [] if (not length or scalar.value is None) else [scalar.value]*length
        return cls(scalar.dtype, _value, scalar.comment, scalar.array_class)

    def __init__(self, dtype, values=None, comment=None, array_class=np.array):
        self.dtype = dtype
        self.comment = comment
        self.array_class = array_class
        self.__set__(None, values)
    def __getitem__(self, key):
        return self._values[key]
    def __setitem__(self, key, value):
        self._values[key] = value
    def __len__(self):
        return self.size

    # Delegate array methods
    def __getattr__(self, attr):
        return getattr(self._values, attr)

    def __repr__(self):
        return "\n".join([self.info, repr(self._values)])
    def __str__(self):
        return self.__repr__()

    @property
    def info(self):
        # _type_ok = "✅" if self.dtype_match else "❌"
        _name = self.__class__.__name__
        return f"{_name} {self.dtype} of len {len(self)}  # {self.comment}"


class Dataset:
    """Base class for typed, annotated datasets."""
    
    @classmethod
    def list_scalars(cls):
        return [k for k,v in cls.__dict__.items() if isinstance(v, Scalar)]

    @classmethod
    def list_vectors(cls):
        return [k for k,v in cls.__dict__.items() if isinstance(v, Vector)]

    def __init__(self, **fields: Union[Scalar|Vector]):
        """Create dataset, dynamically setting field values.
        
        Vectors are initialized first, ensuring all are of equal length.
        Scalars are filled in afterwards.
        """
        
        _sizes = {}
        _vectors = self.__class__.list_vectors()
        if not _vectors:
            raise ValueError("no vectors in your dataset")
        for vector_name in _vectors:
            field_data = fields.get(vector_name)
            setattr(self, vector_name, field_data)
            _sizes[vector_name] = len(self.__dict__[vector_name])
        _max_size = max(_sizes.values())
        if not all([_max_size == v for v in _sizes.values()]):
            raise ValueError(f"vectors must be of equal size: {_sizes}")

        _scalars = self.__class__.list_scalars()
        for scalar_name in _scalars:
            _value = fields.get(scalar_name)
            _scalar = self.__class__.__dict__[scalar_name]
            _scalar.value = _value
            _vector_from_scalar = Vector.from_scalar(_scalar, len(self))
            setattr(self, scalar_name, _vector_from_scalar)

    def __len__(self):
        return len(next(iter(self.__dict__.values())))

    @property
    def data_dict(self):
        return {key: self.__class__.__dict__[key].comment for key in self.__dict__}

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.__dict__)
