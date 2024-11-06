from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import affinity as af

# https://github.com/duckdb/duckdb/issues/14179
duckdb.sql("SET python_scan_all_frames=true")


def test_location_default():
    loc = af.Location()
    assert loc.path == "./export.csv"


def test_location_partitioned():
    loc = af.Location(folder="s3://affinity", partition_by=list("ab"))
    assert loc.path == "s3://affinity/a={}/b={}/export.csv"


def test_scalar():
    s = af.ScalarObject("field comment")
    assert repr(s) == "ScalarObject <class 'object'>  # field comment"


def test_empty_vector_no_dtype():
    with pytest.raises(TypeError):
        af.Vector()


def test_empty_vector():
    v = af.Vector(np.int8)
    assert len(v) == 0
    assert (
        repr(v) == "Vector <class 'numpy.int8'>  # None | len 0\narray([], dtype=int8)"
    )


def test_typed_descriptors():
    s_untyped = af.ScalarObject("")
    assert s_untyped.dtype == object
    v_untyped = af.VectorObject("")
    assert v_untyped.dtype == object
    v_bool = af.VectorBool("")
    assert v_bool.dtype == "boolean"
    v_i8 = af.VectorI8("")
    assert v_i8.dtype == pd.Int8Dtype()
    v_i16 = af.VectorI16("")
    assert v_i16.dtype == pd.Int16Dtype()
    v_i32 = af.VectorI32("")
    assert v_i32.dtype == pd.Int32Dtype()
    v_i64 = af.VectorI64("")
    assert v_i64.dtype == pd.Int64Dtype()
    v_f16 = af.VectorF16("")
    assert v_f16.dtype == np.float16
    v_f32 = af.VectorF32("")
    assert v_f32.dtype == np.float32
    v_f64 = af.VectorF64("")
    assert v_f64.dtype == np.float64
    assert v_f64.__class__.__name__ == "VectorF64"


def test_vector_from_scalar():
    s = af.ScalarBool("single boolean", value=1)
    v = af.Vector.from_scalar(s)
    assert len(v) == 1
    assert v.scalar == 1


def test_dataset_no_attributes():
    class aDataset(af.Dataset):
        pass

    with pytest.raises(ValueError):
        aDataset()


def test_wrong_dataset_declaration():
    class aDataset(af.Dataset):
        v: af.Vector(np.int8)  # type: ignore
        # v = af.Vector(np.int8)  # the correct way

    with pytest.raises(ValueError):
        aDataset()


def test_dataset_with_overflows():
    class aDataset(af.Dataset):
        v = af.Vector(np.int8)

    with pytest.raises(OverflowError):
        aDataset(v=[999])


def test_empty_dataset():
    class aDataset(af.Dataset):
        s = af.ScalarObject("scalar")
        v = af.Vector(np.int8, comment="vector")

    assert repr(aDataset) == "\n".join(
        [
            "aDataset",
            "s: ScalarObject <class 'object'>  # scalar",
            "v: Vector <class 'numpy.int8'>  # vector",
        ]
    )
    data = aDataset()
    assert data.is_dataset("v") is False
    data.alias = "this adds a new key to data.__dict__ but not to data.dict"
    assert data.df.shape == (0, 2)
    assert data.df.dtypes["v"] == np.int8


def test_dataset_instantiation_leaves_class_attrs_unmodified():
    class aDataset(af.Dataset):
        v = af.Vector(np.int8)

    data = aDataset(v=[42])
    assert len(data.v) == 1
    assert len(aDataset.v) == 0


def test_dataset_scalar():
    class aScalarDataset(af.Dataset):
        v1 = af.Scalar(np.bool_, comment="first")
        v2 = af.ScalarF32("second")

    data = aScalarDataset(v1=0, v2=float("-inf"))
    assert not data.v1[-1]
    assert data.v2.dtype == np.float32
    assert data._scalars == dict(v1=0, v2=float("-inf"))
    empty_scalar_dataset_df = aScalarDataset().df
    assert empty_scalar_dataset_df.dtypes.to_list() == [np.bool_, np.float32]


def test_dataset_with_none():
    class aDatasetWithNones(af.Dataset):
        v1 = af.ScalarBool("first")
        v2 = af.VectorI8("second")

    data = aDatasetWithNones(v1=None, v2=[None, 1])
    assert data.shape == (2, 2)


def test_dataset_scalar_vector():
    class aDatasetVectorScalar(af.Dataset):
        """A well-documented dataset."""

        v1 = af.Vector(np.str_, comment="first")
        v2 = af.Scalar(np.int8, comment="second")
        v3 = af.VectorF16("third")

    data1 = aDatasetVectorScalar(v1=list("abcdef"), v2=2, v3=range(6))
    assert len(data1) == 6
    assert data1.shape == (6, 3)
    assert list(data1.v3)[-1] == 5.0
    assert data1.data_dict == {"v1": "first", "v2": "second", "v3": "third"}
    expected_dict = dict(v1=list("abcdef"), v2=2, v3=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert data1.dict == expected_dict
    expected_repr = "\n".join(
        [
            "Dataset aDatasetVectorScalar of shape (6, 3)",
            "v1 = ['a', 'b' ... 'e', 'f']",
            "v2 = 2",
            "v3 = [0.0, 1.0 ... 4.0, 5.0]",
        ]
    )
    assert repr(data1) == expected_repr
    assert data1.metadata.get("table_comment") == "A well-documented dataset."
    assert data1.metadata.get("source") == "manual"
    expected_df = pd.DataFrame({"v1": list("abcdef"), "v2": 2, "v3": range(6)}).astype(
        {"v1": np.str_, "v2": np.int8, "v3": np.float16}
    )
    pd.testing.assert_frame_equal(data1.df, expected_df)

    class aDatasetOnlyVector(af.Dataset):
        v1 = af.Vector(np.str_, comment="first")
        v2 = af.Vector(np.int8, comment="second")
        v3 = af.VectorF16("third")

    data2 = aDatasetOnlyVector(v1=list("abcdef"), v2=[2] * 6, v3=[0, 1, 2, 3, 4, 5])
    pd.testing.assert_frame_equal(data1.df, data2.df)
    assert data1 == data2


def test_from_dataframe():
    class aDataset(af.Dataset):
        v1 = af.VectorBool("")
        v2 = af.VectorF32("")
        v3 = af.VectorI16("")

    source_df = pd.DataFrame(
        {
            "v1": [1, 0],
            "v2": [0.0, 1.0],
            "v3": [None, -1],
        }
    )
    data = aDataset.build(dataframe=source_df)
    data2 = aDataset.from_dataframe(source_df)
    pd.testing.assert_frame_equal(data.df, data2.df)
    assert data.origin.get("source") == "dataframe, shape (2, 3)"
    default_dtypes = source_df.dtypes
    desired_dtypes = {"v1": "boolean", "v2": np.float32, "v3": pd.Int16Dtype()}
    pd.testing.assert_frame_equal(data.df, source_df.astype(desired_dtypes))
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(data.df, source_df.astype(default_dtypes))


def test_from_query():
    class aDataset(af.Dataset):
        v1 = af.VectorBool("")
        v2 = af.VectorF32("")
        v3 = af.VectorI16("")

    source_df = pd.DataFrame(
        {
            "v1": [1, 0],
            "v2": [0.0, 1.0],
            "v3": [None, -1],
        }
    )
    data = aDataset.build(query="FROM source_df")
    assert (
        data.origin.get("source") == "dataframe, shape (2, 3)\nquery:\nFROM source_df"
    )
    default_dtypes = source_df.dtypes
    desired_dtypes = {"v1": "boolean", "v2": np.float32, "v3": pd.Int16Dtype()}
    pd.testing.assert_frame_equal(data.df, source_df.astype(desired_dtypes))
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(data.df, source_df.astype(default_dtypes))


def test_to_polars():
    class aDataset(af.Dataset):
        v1 = af.VectorBool("")
        v2 = af.VectorF32("")
        v3 = af.VectorI16("")

    data = aDataset(v1=[True], v2=[1 / 2], v3=[999])
    polars_df = data.pl
    assert str(polars_df.dtypes) == "[Boolean, Float32, Int16]"


def test_to_pyarrow():
    class aDataset(af.Dataset):
        v1 = af.VectorBool("")
        v2 = af.VectorF32("")
        v3 = af.VectorI16("")

    data = aDataset(v1=[True], v2=[1 / 2], v3=[999])
    arrow_table = data.arrow
    assert all(
        key in arrow_table.schema.metadata.keys() for key in [b"v1", b"v2", b"v3"]
    )


def test_sql_simple():
    class aDataset(af.Dataset):
        v1 = af.VectorI8("")
        v2 = af.VectorBool("")

    data_a = aDataset(v1=[1, 2], v2=[True, False])
    data_a_sql_df = data_a.sql("FROM df").df()
    assert (data_a_sql_df.values == data_a.df.values).all()


def test_sql_join():
    class aDataset(af.Dataset):
        v1 = af.VectorI8("")
        v2 = af.VectorBool("")

    data_a = aDataset(v1=[1, 2], v2=[True, False])

    class bDataset(af.Dataset):
        v1 = af.VectorI8("")
        v3 = af.VectorObject("")

    data_b = bDataset(v1=[1, 3], v3=["foo", "moo"])
    joined = data_a.sql("FROM df JOIN dfb USING (v1)", dfb=data_b.pl)
    assert joined.fetchone() == (1, True, "foo")


def test_replacement_scan_persistence_from_last_test():
    class cDataset(af.Dataset):
        v1 = af.VectorI8("")

    cDataset().sql("FROM dfb")  # "dfb" from last test still available
    with pytest.raises(Exception):
        cDataset().sql("SELECT v2 FROM df")  # "df" != last test's data_a.df


def test_to_parquet_with_metadata():
    class aDataset(af.Dataset):
        """Delightful data."""

        v1 = af.VectorBool(comment="is that so?")
        v2 = af.VectorF32(comment="float like a butterfly")
        v3 = af.VectorI16(comment="int like a three")

    data = aDataset(v1=[True], v2=[1 / 2], v3=[3])
    test_file_arrow = Path("test_arrow.parquet")
    test_file_duckdb = Path("test_duckdb.parquet")
    test_file_duckdb_polars = Path("test_duckdb_polars.parquet")
    data.to_parquet(test_file_arrow, engine="arrow")
    data.to_parquet(test_file_duckdb, engine="duckdb")
    data.to_parquet(test_file_duckdb_polars, engine="duckdb", df=data.pl)

    class KeyValueMetadata(af.Dataset):
        """Stores results of reading Parquet metadata."""

        key = af.VectorObject("")
        value = af.VectorObject("")

    test_file_metadata_arrow = KeyValueMetadata.from_sql(
        f"""
        SELECT
            file_name,
            DECODE(key) AS key,
            DECODE(value) AS value,
        FROM parquet_kv_metadata('{test_file_arrow}')
        WHERE DECODE(key) != 'ARROW:schema'
        """,
        method="pandas",
        field_names="strict",
    )
    test_file_metadata_duckdb = KeyValueMetadata.from_sql(
        f"""
        SELECT
            DECODE(key) AS key,
            DECODE(value) AS value,
        FROM parquet_kv_metadata('{test_file_duckdb_polars}')
        WHERE DECODE(key) != 'ARROW:schema'
        """,
        method="polars",
        field_names="strict",
    )
    assert test_file_metadata_arrow == test_file_metadata_duckdb
    test_file_arrow.unlink()
    test_file_duckdb.unlink()
    test_file_duckdb_polars.unlink()
    assert all(
        value in test_file_metadata_arrow.value.values
        for value in [
            "is that so?",
            "float like a butterfly",
            "int like a three",
            "Delightful data.",
            "manual",
        ]
    )


def test_parquet_roundtrip_with_rename():
    class IsotopeData(af.Dataset):
        symbol = af.VectorObject("Element")
        z = af.VectorI8("Atomic Number (Z)")
        mass = af.VectorF64("Isotope Mass (Da)")
        abundance = af.VectorF64("Relative natural abundance")

    url = "https://raw.githubusercontent.com/liquidcarbon/chembiodata/main/isotopes.csv"
    with pytest.raises(KeyError):
        IsotopeData.build(query=f"FROM '{url}'")
    data_from_sql = IsotopeData.build(query=f"FROM '{url}'", rename=True)
    assert len(data_from_sql) == 354
    test_file = Path("test.parquet")
    data_from_sql.to_parquet(test_file, engine="arrow")
    data_from_parquet_arrow = IsotopeData.build(query=f"FROM '{test_file}'")
    data_from_sql.to_parquet(test_file, engine="duckdb")
    data_from_parquet_duckdb = IsotopeData.build(query=f"FROM '{test_file}'")
    test_file.unlink()
    assert data_from_sql == data_from_parquet_arrow
    assert data_from_parquet_duckdb == data_from_parquet_arrow


def test_partition():
    class aDataset(af.Dataset):
        v1 = af.VectorObject(comment="partition")
        v2 = af.VectorI16(comment="int like a three")
        v3 = af.VectorF32(comment="float like a butterfly")

    adata = aDataset(v1=list("aaabbc"), v2=[1, 2, 1, 2, 1, 2], v3=[9, 8, 7, 7, 8, 9])

    names, folders, filepaths, datasets = adata.partition()
    assert filepaths[0] == "./aDataset_export.csv"
    assert datasets[0] == adata

    adata.LOCATION.folder = "test_save"
    adata.LOCATION.partition_by = ["v1", "v2"]
    names, folders, filepaths, datasets = adata.partition()
    assert names == [["a", "1"], ["a", "2"], ["b", "1"], ["b", "2"], ["c", "2"]]
    assert folders[-1] == "test_save/v1=c/v2=2/"
    assert len(filepaths) == 5
    assert [len(p) for p in datasets] == [2, 1, 1, 1, 1]

    class bDataset(af.Dataset):
        v1 = af.VectorObject(comment="partition")
        v2 = af.VectorI16(comment="int like a three")
        v3 = af.VectorF32(comment="float like a butterfly")
        LOCATION = af.Location(folder="s3://mybucket/affinity/", partition_by=["v1"])

    bdata = bDataset.build(dataframe=adata.df)
    names, folders, filepaths, datasets = bdata.partition()
    assert filepaths[0] == "s3://mybucket/affinity/v1=a/export.csv"


def test_flatten_nested_dataset():
    class User(af.Dataset):
        name = af.ScalarObject("username")
        attrs = af.VectorObject("user attributes")

    class Task(af.Dataset):
        created_ts = af.ScalarF64("created timestamp")
        user = af.VectorObject("user")
        hours = af.VectorI16("time worked (hours)")

    u1 = User(name="Alice", attrs=["adorable", "agreeable"])
    u2 = User(name="Brent", attrs=["bland", "broke"])
    t1 = Task(created_ts=123.456, user=[u1, u2], hours=[3, 5])
    assert t1.is_dataset("user") is True
    assert t1.is_dataset("qty") is False
    expected_dict = {
        "created_ts": 123.456,
        "user": [
            {"name": "Alice", "attrs": ["adorable", "agreeable"]},
            {"name": "Brent", "attrs": ["bland", "broke"]},
        ],
        "hours": [3, 5],
    }
    assert t1.model_dump() == expected_dict
    flattened_df = pd.DataFrame(
        {
            "created_ts": 123.456,
            "name": ["Alice", "Brent"],
            "attrs": [["adorable", "agreeable"], ["bland", "broke"]],
            "hours": [3, 5],
        },
    )
    assert t1.flatten(prefix=False).to_dict() == flattened_df.to_dict()
    assert set(t1.flatten(prefix=True).columns) == {
        "created_ts",
        "hours",
        "user.name",
        "user.attrs",
    }
