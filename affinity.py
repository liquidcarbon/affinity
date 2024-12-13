__doc__ = """
Module for creating well-documented datasets, with types and annotations.
"""

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd


class _modules:
    """Stores modules imported conditionally."""

    def try_import(modules: List[str]) -> None:
        """Conditional imports."""
        for module in modules:
            try:
                _module = import_module(module)
                globals()[module] = _module  # used here
                setattr(_modules, module, _module)  # used in tests
            except ImportError:
                setattr(_modules, module, False)


if TYPE_CHECKING:
    import awswrangler  # type: ignore
    import polars  # type: ignore
    import pyarrow  # type: ignore
    import pyarrow.parquet  # type: ignore
else:
    _modules.try_import(["awswrangler", "polars", "pyarrow", "pyarrow.parquet"])


@dataclass
class Location:
    """Dataclass for writing the data.

    Used in a special attribute `Dataset.LOCATION` that is attached to
    all Datasets via metaclass by default, or can be set explicitly.
    """

    folder: str | Path = field(default=Path("."))
    file: str | Path = field(default="export.csv")
    partition_by: List[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        """Generates paths for writing partitioned data."""
        _path = (
            self.folder.as_posix() if isinstance(self.folder, Path) else self.folder
        ).rstrip("/")
        for part in self.partition_by:
            _path += f"/{part}={{}}"
        else:
            _path += f"/{self.file}"
        return _path


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
        except Exception as e:
            # blanket exception to troubleshoot, thus far it never came up
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

        I failed to convince IDEs that factory-made classes are not of "DescriptorType"
        and reverted to explicit class declarations.  Keeping for posterity.
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
    """Metaclass for universal attributes and custom repr."""

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        if "LOCATION" not in dct:
            new_class.LOCATION = Location(file=f"{name}_export.csv")
        return new_class

    def __repr__(cls) -> str:
        _lines = [cls.__name__]
        for k, v in cls.__dict__.items():
            if isinstance(v, Descriptor):
                _lines.append(f"{k}: {v.info}")
            if isinstance(v, DatasetMeta):
                _lines.append(f"{k}: {v.__doc__}")
        return "\n".join(_lines)


class DatasetBase(metaclass=DatasetMeta):
    """Parent class and classmethods for main Dataset class."""

    @classmethod
    def as_field(cls, as_type: str | Scalar | Vector = Vector, comment: str = ""):
        _comment = comment or cls.__doc__
        if as_type in (Scalar, "scalar"):
            return ScalarObject(_comment)
        elif as_type in (Vector, "vector"):
            return VectorObject(_comment)

    @classmethod
    def get_scalars(cls):
        return {k: None for k, v in cls.__dict__.items() if isinstance(v, Scalar)}

    @classmethod
    def get_vectors(cls):
        return {k: v for k, v in cls.__dict__.items() if isinstance(v, Vector)}

    @classmethod
    def get_dict(cls):
        return dict(cls())

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
        cls, dataframe: pd.DataFrame | Optional["polars.DataFrame"], **kwargs
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

    @property
    def athena_types(self):
        """Convert pandas types to SQL types for loading into AWS Athena."""
        columns_types, partition_types = awswrangler.catalog.extract_athena_types(
            df=self.df,
            partition_cols=self.LOCATION.partition_by,
        )
        return columns_types, partition_types

    def kwargs_for_create_athena_table(
        self, db: str, table: str, compression: str | None = None, **kwargs
    ):
        """Arguments for creating AWS Athena tables."""
        columns_types, partitions_types = self.athena_types
        return dict(
            database=db,
            table=table,
            path=self.LOCATION.folder,
            columns_types=columns_types,
            partitions_types=partitions_types,
            compression=compression,
            description=self.__doc__,
            columns_comments=self.data_dict,
            **kwargs,
        )


class Dataset(DatasetBase):
    """Base class for typed, annotated datasets."""

    def __init__(self, **fields: Scalar | Vector):
        """Create dataset, dynamically setting field values.

        Vectors are initialized first, ensuring all are of equal length.
        Scalars are filled in afterwards.
        """
        self._vectors = self.__class__.get_vectors()
        self._scalars = self.__class__.get_scalars()
        if len(self._vectors) + len(self._scalars) == 0:
            raise ValueError("no attributes defined in your dataset")
        self.origin = {"created_ts": int(time() * 1000)}
        _sizes = {}
        for vector_name in self._vectors:
            _values = fields.get(vector_name)
            setattr(self, vector_name, _values)
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
            if isinstance(_value, Dataset):
                self._scalars[scalar_name] = _value.dict
            else:
                self._scalars[scalar_name] = _value

        if len(self.origin) == 1:  # only after direct __init__
            self.origin["source"] = "manual"

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
        n_cols = len(self._vectors) + len(self._scalars)
        return len(self), n_cols

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
    def arrow(self) -> "pyarrow.Table":
        metadata = {str(k): str(v) for k, v in self.metadata.items()}
        _dict = {
            k: [v.dict for v in vector] if self.is_dataset(k) else vector
            for k, vector in self
        }
        return pyarrow.table(_dict, metadata=metadata)

    @property
    def pl(self) -> "polars.DataFrame":
        return polars.DataFrame(dict(self))

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
        TODO: add registrations to `from_sql`
        """
        if replacements.get("df") is None:
            duckdb.register("df", self.df)
        for k, v in replacements.items():
            duckdb.register(k, v)
        return duckdb.sql(query)

    def flatten(self, prefix: bool = False) -> pd.DataFrame:
        """Returns a flattened dataset. Experimental.

        With prefix=False, columns in flattened data will be named and ordered
        as they appear in child attributes.  With prefix=False, the names will be
        dotted paths (user.name), and the order may be different.
        """
        if prefix:
            return pd.json_normalize(self.df.to_dict("records"))
        else:
            return pd.concat(
                [
                    pd.json_normalize(self.df[col])
                    if isinstance(getattr(self, col)[0], Dataset)
                    else self.df[col]
                    for col in self.df
                ],
                axis=1,
            )

    def model_dump(self) -> dict:
        """Similar to Pydantic's model_dump; alias for dict."""
        return self.dict

    def to_parquet(self, path, engine="duckdb", **kwargs):
        if engine == "pandas":
            self.df.to_parquet(path)
        elif engine == "arrow":
            pyarrow.parquet.write_table(self.arrow, path)
        elif engine == "duckdb":
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
        else:
            raise NotImplementedError
        return path

    def partition(self) -> Tuple[List[str], List[str], List[str], List[DatasetBase]]:
        """Path and format constructed from `LOCATION` attribute.

        Variety of outputs is helpful when populating cloud warehouses,
        such as Athena/Glue via awswrangler.
        """

        _file = Path(self.LOCATION.file)
        _stem = _file.stem
        _ext = _file.suffix
        if len(self.LOCATION.partition_by) == 0:
            _partitions_iter = zip([""], [self.df])
        else:
            _partitions_iter = self.df.groupby(self.LOCATION.partition_by)
        names = []
        folders = []
        filepaths = []
        datasets = []
        for name, data in _partitions_iter:
            _path = self.LOCATION.path.format(*name)
            names.append([str(p) for p in name])
            folders.append(_path.rsplit("/", maxsplit=1)[0] + "/")
            filepaths.append(_path)
            datasets.append(self.__class__.build(dataframe=data))
        return names, folders, filepaths, datasets


### Typed scalars and vectors.  TODO: datetimes?


class ScalarObject(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(object, value, comment, array_class)


class ScalarString(Scalar):
    def __init__(self, comment: str, *, value=None, array_class=pd.Series):
        super().__init__(pd.StringDtype(), value, comment, array_class)


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


class VectorString(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(pd.StringDtype(), values, comment, array_class)


class VectorBool(Vector):
    def __init__(self, comment: str, *, values=None, array_class=pd.Series):
        super().__init__(np.bool, values, comment, array_class)


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
