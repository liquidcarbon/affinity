__doc__ = """
Module for creating well-documented datasets, with types and annotations.
"""

from importlib import import_module
from time import time
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd


def try_import(module) -> Optional[object]:
    try:
        return import_module(module)
    except ImportError:
        print(f"{module} not found in the current environment")
        return


if TYPE_CHECKING:
    import duckdb  # type: ignore
    import polars as pl  # type: ignore
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
else:
    duckdb = try_import("duckdb")
    pl = try_import("polars")
    pa = try_import("pyarrow")
    pq = try_import("pyarrow.parquet")


class Descriptor:
    """Base class for scalars and vectors."""

    def __get__(self, instance, owner):
        return self if not instance else instance.__dict__[self.name]

    def __set__(self, instance, values):
        try:
            _values = self.array_class(
                values if values is not None else [], dtype=self.dtype
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

    @property
    def info(self):
        _name = self.__class__.__name__
        return f"{_name} {self.dtype}  # {self.comment}"

    @classmethod
    def factory(cls, dtype, array_class=pd.Series, cls_name=None):
        """Factory method for creating typed classes.

        Reverted to explicit class declarations.
        Unable to convince IDEs that factory-made classes are not of "DescriptorType".
        """

        class DescriptorType(cls):
            def __init__(self, comment=None, *, values=None, array_class=array_class):
                super().__init__(dtype, values, comment, array_class)

        if cls_name:
            DescriptorType.__name__ = cls_name
        return DescriptorType


class Scalar(Descriptor):
    """Scalar is a single value. In datasets, it's repeated len(dataset) times."""

    def __init__(self, dtype, value=None, comment=None, array_class=np.array):
        self.dtype = dtype
        self.value = value
        self.comment = comment
        self.array_class = array_class

    def __len__(self):
        return 1

    def __repr__(self):
        return self.info


class Vector(Descriptor):
    """Vectors are typed arrays of values."""

    @classmethod
    def from_scalar(cls, scalar: Scalar, length=1):
        _value = [] if (not length or scalar.value is None) else [scalar.value] * length
        instance = cls(scalar.dtype, _value, scalar.comment, scalar.array_class)
        instance.scalar = scalar.value
        return instance

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
        return "\n".join([f"{self.info} | len {len(self)}", repr(self._values)])

    def __str__(self):
        return self.__repr__()


class DatasetMeta(type):
    """Metaclass for custom repr."""

    def __repr__(cls) -> str:
        _lines = [cls.__name__]
        for k, v in cls.__dict__.items():
            if isinstance(v, Descriptor):
                _lines.append(f"{k}: {v.info}")
        return "\n".join(_lines)


class Dataset(metaclass=DatasetMeta):
    """Base class for typed, annotated datasets."""

    save_to = {"partition": tuple(), "prefix": "", "file": ""}

    @classmethod
    def get_scalars(cls):
        return {k: None for k, v in cls.__dict__.items() if isinstance(v, Scalar)}

    @classmethod
    def get_vectors(cls):
        return {k: None for k, v in cls.__dict__.items() if isinstance(v, Vector)}

    @classmethod
    def get_dict(cls):
        return dict(cls())

    def __init__(self, **fields: Union[Scalar, Vector]):
        """Create dataset, dynamically setting field values.

        Vectors are initialized first, ensuring all are of equal length.
        Scalars are filled in afterwards.
        """

        self._vectors = self.__class__.get_vectors()
        self._scalars = self.__class__.get_scalars()
        if len(self._vectors) == 0 and len(self._scalars) == 0:
            raise ValueError("no attributes defined in your dataset")
        self.origin = {"created_ts": int(time() * 1000)}
        _sizes = {}
        for vector_name in self._vectors:
            field_data = fields.get(vector_name)
            setattr(self, vector_name, field_data)
            _sizes[vector_name] = len(self.__dict__[vector_name])
        if len(self._vectors) > 0:
            self._max_size = max(_sizes.values())
            if not all([self._max_size == v for v in _sizes.values()]):
                raise ValueError(f"vectors must be of equal size: {_sizes}")
        else:
            self._max_size = 1

        for scalar_name in self._scalars:
            _value = fields.get(scalar_name)
            _scalar = self.__class__.__dict__[scalar_name]
            _scalar.value = _value
            _vector_from_scalar = Vector.from_scalar(_scalar, self._max_size)
            setattr(self, scalar_name, _vector_from_scalar)
            self._scalars[scalar_name] = _value

        if len(self.origin) == 1:  # only after direct __init__
            self.origin["source"] = "manual"

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
    def from_dataframe(
        cls, dataframe: pd.DataFrame | Optional["pl.DataFrame"], **kwargs
    ):
        instance = cls()
        for i, k in enumerate(dict(instance)):
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

    def __eq__(self, other):
        return self.df.equals(other.df)

    def __len__(self) -> int:
        return max(len(field[1]) for field in self)

    def __iter__(self):
        """Yields attr names and values, in same order as defined in class."""
        yield from (
            (k, self.__dict__[k]) for k in self.__class__.__dict__ if k in self.__dict__
        )

    def __repr__(self):
        lines = [f"Dataset {self.__class__.__name__} of shape {self.shape}"]
        dict_list = self.df4.to_dict("list")
        dict_list.update(**self._scalars)
        for k, v in dict_list.items():
            lines.append(f"{k} = {v}".replace(", '...',", " ..."))
        return "\n".join(lines)

    @property
    def shape(self):
        return len(self), len(self._vectors) + len(self._scalars)

    @property
    def dict(self) -> dict:
        """JSON-like dict, with scalars as scalars and vectors as lists."""
        _dict = self.df.to_dict("list")
        return {**_dict, **self._scalars}

    @property
    def data_dict(self) -> dict:
        return {k: self.__class__.__dict__[k].comment for k, v in self}

    @property
    def metadata(self) -> dict:
        """The metadata for the dataclass instance."""
        return {
            "table_comment": self.__class__.__doc__,
            **self.data_dict,
            **self.origin,
        }

    @property
    def df(self) -> pd.DataFrame:
        _dict = {
            k: [v.dict for v in vector] if self.is_dataset(k) else vector
            for k, vector in self
        }
        return pd.DataFrame(_dict)

    @property
    def df4(self) -> pd.DataFrame:
        if len(self) > 4:
            df = self.df.iloc[[0, 1, -2, -1], :]
            df.loc[1.5] = "..."  # fake spacer row
            return df.sort_index()
        else:
            return self.df

    @property
    def arrow(self) -> "pa.Table":
        metadata = {str(k): str(v) for k, v in self.metadata.items()}
        _dict = {
            k: [v.dict for v in vector] if self.is_dataset(k) else vector
            for k, vector in self
        }
        return pa.table(_dict, metadata=metadata)

    @property
    def pl(self) -> "pl.DataFrame":
        return pl.DataFrame(dict(self))

    def is_dataset(self, key):
        attr = getattr(self, key, None)
        if attr is None or len(attr) == 0 or isinstance(attr, Scalar):
            return False
        else:
            return all(isinstance(v, Dataset) for v in attr)

    def sql(self, query, **replacements):
        """Query the dataset with DuckDB.

        DuckDB uses replacement scans to query python objects.
        Class instance attributes like `FROM self.df` must be registered as views.
        This is what **replacements kwargs are for.
        By default, df=self.df (pandas dataframe) is used.
        The registered views persist across queries.  RAM impact TBD.
        """
        if replacements.get("df") is None:
            duckdb.register("df", self.df)
        for k, v in replacements.items():
            duckdb.register(k, v)
        return duckdb.sql(query)

    def flatten(self):
        """List of dicts? Dict of lists? TBD"""
        raise NotImplementedError

    def model_dump(self) -> dict:
        """Similar to Pydantic's model_dump; alias for dict."""
        return self.dict

    def to_parquet(self, path, engine="duckdb", **kwargs):
        if engine == "arrow":
            pq.write_table(self.arrow, path)
        if engine == "duckdb":
            kv_metadata = []
            for k, v in self.metadata.items():
                if isinstance(v, str) and "'" in v:
                    _v = {v.replace("'", "''")}  # must escape single quotes
                    kv_metadata.append(f"{k}: '{_v}'")
                else:
                    kv_metadata.append(f"{k}: '{v}'")
            self.sql(
                f"""
            COPY (SELECT * FROM df) TO {path} (
                FORMAT PARQUET,
                KV_METADATA {{ {", ".join(kv_metadata)} }}
            );""",
                **kwargs,
            )
        return path

    def save(self):
        """Path and format constructed from `save_to` attribute."""
        raise NotImplementedError


### Typed scalars and vectors


class ScalarObject(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(object, value, comment, array_class)


class ScalarBool(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__("boolean", value, comment, array_class)


class ScalarI8(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(pd.Int8Dtype(), value, comment, array_class)


class ScalarI16(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(pd.Int16Dtype(), value, comment, array_class)


class ScalarI32(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(pd.Int32Dtype(), value, comment, array_class)


class ScalarI64(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(pd.Int64Dtype(), value, comment, array_class)


class ScalarF32(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(np.float32, value, comment, array_class)


class ScalarF64(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(np.float64, value, comment, array_class)


class VectorObject(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(object, values, comment, array_class)


class VectorBool(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__("boolean", values, comment, array_class)


class VectorI8(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int8Dtype(), values, comment, array_class)


class VectorI16(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int16Dtype(), values, comment, array_class)


class VectorI32(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int32Dtype(), values, comment, array_class)


class VectorI64(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(pd.Int64Dtype(), values, comment, array_class)


class VectorF16(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(np.float16, values, comment, array_class)


class VectorF32(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(np.float32, values, comment, array_class)


class VectorF64(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(np.float64, values, comment, array_class)
