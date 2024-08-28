__doc__ = """
Module for creating well-documented datasets, with types and annotations.
"""

import numpy as np
import pandas as pd
from importlib import import_module
from time import time
from typing import TYPE_CHECKING, Optional, Union


def try_import(module) -> Optional[object]:
    try:
        return import_module(module)
    except ImportError:
        print(f"{module} not found in the current environment")
        return

if TYPE_CHECKING:
    import duckdb  # type: ignore
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    import polars as pl  # type: ignore
else:
    duckdb = try_import("duckdb")
    pl = try_import("polars")      
    pa = try_import("pyarrow")
    pq = try_import("pyarrow.parquet")


class Descriptor:
    def __get__(self, instance, owner):
        return self if not instance else instance.__dict__[self.name]
    
    def __set__(self, instance, values):
        try:
            _values = self.array_class(
                values if values is not None else [],
                dtype=self.dtype
            )
        except OverflowError as e:
            raise e
        except Exception as e:  # leaving blanket exception to troubleshoot
            raise e
        if instance is None:
            self._values = _values
        else:
            instance.__dict__[self.name] = _values

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
        # if attr in ("_values",):
        #     return self._values
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
        
        self.origin = {"created_ts": int(time() * 1000)}
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
        
        if len(self.origin) == 1:  # only after direct __init__
            self.origin["source"] = "manual"

    def __eq__(self, other):
        return self.df.equals(other.df)
    
    @classmethod
    def build(cls, query=None, dataframe=None, **kwargs):
        """Build from DuckDB query or a dataframe.
        
        Build kwargs:
        - rename: how to handle source with differently named fields:
          None|False: field names in source must match class declaration
          True: fields in source fetched, renamed in same order they're declared
        """
        if query:
            return cls.from_sql(query, **kwargs)
        if isinstance(dataframe, (pd.DataFrame,)):
            return cls.from_dataframe(dataframe, **kwargs)
    
    @classmethod
    def from_dataframe(cls, dataframe: Union[pd.DataFrame|pl.DataFrame], **kwargs):
        instance = cls()
        for i, k in enumerate(instance.dict):
            if kwargs.get("rename") in (None, False):
                setattr(instance, k, dataframe[k])
            else:
                setattr(instance, k, dataframe[dataframe.columns[i]])
        instance.origin["source"] = f"dataframe, shape {dataframe.shape}"
        return instance
    
    @classmethod
    def from_sql(cls, query: str, **kwargs):
        if kwargs.get("method") in (None, "pandas"):
            query_results = duckdb.sql(query).df()
        if kwargs.get("method") in ("polars",):
            query_results = duckdb.sql(query).pl()        
        instance = cls.from_dataframe(query_results, **kwargs)
        instance.origin["source"] += f"\nquery:\n{query}"
        return instance

    def __len__(self) -> int:
        return len(next(iter(self.dict.values())))
    
    def sql(self, query):
        """Out of scope. DuckDB won't let `FROM self.df`, must register views."""
        raise NotImplementedError
    
    def to_parquet(self, path):
        pq.write_table(self.arrow, path)
        return

    @property
    def dict(self) -> dict:
        """Distinct from __dict__: only includes attributes defined in the class."""
        return {
            k: v for k, v in self.__dict__.items() if k in self.__class__.__dict__
        }

    @property
    def data_dict(self) -> dict:
        return {k: self.__class__.__dict__[k].comment for k in self.dict}

    @property
    def metadata(self) -> dict:
        """The metadata for the dataclass instance."""
        return {
            "table_comment": self.__class__.__doc__,
            **self.data_dict,
            **self.origin
        }

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.dict)

    @property
    def arrow(self) -> "pa.Table":
        metadata = {str(k): str(v) for k, v in self.metadata.items()}
        return pa.table(self.dict, metadata=metadata)

    @property
    def pl(self) -> "pl.DataFrame":
        return pl.DataFrame(self.dict)


class VectorUntyped(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(object, values, comment, array_class)

class VectorI8(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int8Dtype(), values, comment, array_class)

class VectorBool(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__("boolean", values, comment, array_class)

class VectorI16(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int16Dtype(), values, comment, array_class)

class VectorI32(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int32Dtype(), values, comment, array_class)

class VectorI64(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int64Dtype(), values, comment, array_class)

class VectorF16(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(np.float16, values, comment, array_class)

class VectorF32(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(np.float32, values, comment, array_class)

class VectorF64(Vector):
    def __init__(self, comment=None, *, values=None, array_class=pd.Series):
        super().__init__(np.float64, values, comment, array_class)
