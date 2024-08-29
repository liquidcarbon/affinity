import affinity as af
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_empty_vector_no_dtype():
    with pytest.raises(TypeError):
        v = af.Vector()


def test_empty_vector():
    v = af.Vector(np.int8)
    assert len(v) == 0
    assert repr(v) == "Vector <class 'numpy.int8'> of len 0  # None\narray([], dtype=int8)"


def test_typed_descriptors():
    s_untyped = af.ScalarObject()
    assert s_untyped.dtype == object
    v_untyped = af.VectorObject()
    assert v_untyped.dtype == object
    v_bool = af.VectorBool()
    assert v_bool.dtype == "boolean"
    v_i8 = af.VectorI8()
    assert v_i8.dtype == pd.Int8Dtype()
    v_i16 = af.VectorI16()
    assert v_i16.dtype == pd.Int16Dtype()
    v_i32 = af.VectorI32()
    assert v_i32.dtype == pd.Int32Dtype()
    v_i64 = af.VectorI64()
    assert v_i64.dtype == pd.Int64Dtype()
    v_f16 = af.VectorF16()
    assert v_f16.dtype == np.float16
    v_f32 = af.VectorF32()
    assert v_f32.dtype == np.float32
    v_f64 = af.VectorF64()
    assert v_f64.dtype == np.float64


def test_vector_from_scalar():
    s = af.ScalarBool("single boolean", values=1)
    v = af.Vector.from_scalar(s)
    assert len(v) == 1
    assert v.scalar == True


def test_dataset_no_attributes():
    class aDataset(af.Dataset):
        pass
    with pytest.raises(ValueError):
        data = aDataset()


def test_wrong_dataset_declaration():
    class aDataset(af.Dataset):
        v: af.Vector(np.int8)  # type: ignore
        # v = af.Vector(np.int8)  # the correct way
    with pytest.raises(ValueError):
        data = aDataset()


def test_dataset_with_overflows():
    class aDataset(af.Dataset):
        v = af.Vector(np.int8)
    with pytest.raises(OverflowError):
        data = aDataset(v=[999])


def test_empty_dataset():
    class aDataset(af.Dataset):
        v = af.Vector(np.int8)
    data = aDataset()
    assert data.is_dataset("v") == False
    data.alias = "this adds a new key to data.__dict__ but not to data.dict"
    assert data.df.shape == (0, 1)
    assert data.df.dtypes["v"] == np.int8


def test_dataset_instantiation_leaves_class_attrs_unmodified():
    class aDataset(af.Dataset):
        v = af.Vector(np.int8)
    data = aDataset(v=[42])
    assert len(aDataset.v) == 0


def test_dataset_scalar():
    class aScalarDataset(af.Dataset):
        v1 = af.Scalar(np.bool_, comment="first")
        v2 = af.ScalarF16("second")
    data = aScalarDataset(v1=0, v2=float("-inf"))
    assert data.v1[-1] == False
    assert data.v2.dtype == np.float16
    assert data._scalars == dict(v1=0, v2=float("-inf"))


def test_dataset_scalar_vector():
    class aDatasetVectorScalar(af.Dataset):
        """A well-documented dataset."""
        v1 = af.Vector(np.str_, comment="first")
        v2 = af.Scalar(np.int8, comment="second")
        v3 = af.VectorF16("third")
    data1 = aDatasetVectorScalar(
        v1 = list("abcdef"),
        v2 = 2,
        v3 = range(6)
    )
    assert len(data1) == 6
    assert data1.shape == (6, 3)
    assert data1.data_dict == {"v1": "first", "v2": "second", "v3": "third"}
    expected_dict = dict(
        v1 = list("abcdef"),
        v2 = 2,
        v3 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert data1.dict == expected_dict
    expected_repr = "\n".join([
        "Dataset aDatasetVectorScalar of shape (6, 3)",
        "v1 = ['a', 'b' ... 'e', 'f']",
        "v2 = 2",
        "v3 = [0.0, 1.0 ... 4.0, 5.0]",
    ])
    assert repr(data1) == expected_repr
    assert data1.metadata.get("table_comment") == "A well-documented dataset."
    assert data1.metadata.get("source") == "manual"
    expected_df = pd.DataFrame({
        "v1": list("abcdef"),
        "v2": 2,
        "v3": range(6)
    }).astype({"v1": np.str_, "v2": np.int8, "v3": np.float16})
    pd.testing.assert_frame_equal(data1.df, expected_df)
    class aDatasetOnlyVector(af.Dataset):
        v1 = af.Vector(np.str_, comment="first")
        v2 = af.Vector(np.int8, comment="second")
        v3 = af.VectorF16("third")
    data2 = aDatasetOnlyVector(
        v1 = list("abcdef"),
        v2 = [2] * 6,
        v3 = [0, 1, 2, 3, 4, 5]
    )
    pd.testing.assert_frame_equal(data1.df, data2.df)
    assert data1 == data2


def test_from_dataframe():
    class aDataset(af.Dataset):
        v1 = af.VectorBool()
        v2 = af.VectorF32()
        v3 = af.VectorI16()
    source_df = pd.DataFrame({
        "v1": [1, 0],
        "v2": [0., 1.],
        "v3": [None, -1],
    })
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
        v1 = af.VectorBool()
        v2 = af.VectorF32()
        v3 = af.VectorI16()
    source_df = pd.DataFrame({
        "v1": [1, 0],
        "v2": [0., 1.],
        "v3": [None, -1],
    })
    data = aDataset.build(query="FROM source_df")
    assert data.origin.get("source") == "dataframe, shape (2, 3)\nquery:\nFROM source_df"
    default_dtypes = source_df.dtypes
    desired_dtypes = {"v1": "boolean", "v2": np.float32, "v3": pd.Int16Dtype()}
    pd.testing.assert_frame_equal(data.df, source_df.astype(desired_dtypes))   
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(data.df, source_df.astype(default_dtypes))


def test_to_polars():
    class aDataset(af.Dataset):
        v1 = af.VectorBool()
        v2 = af.VectorF32()
        v3 = af.VectorI16()
    data = aDataset(v1=[True], v2=[1/2], v3=[999])
    polars_df = data.pl
    assert str(polars_df.dtypes) == "[Boolean, Float32, Int16]"


def test_to_pyarrow():
    class aDataset(af.Dataset):
        v1 = af.VectorBool()
        v2 = af.VectorF32()
        v3 = af.VectorI16()
    data = aDataset(v1=[True], v2=[1/2], v3=[999])
    arrow_table = data.arrow
    assert all(
        key in arrow_table.schema.metadata.keys()
        for key in [b"v1", b"v2", b"v3"]
    )


def test_to_parquet_with_metadata():
    class aDataset(af.Dataset):
        """Delightful data."""
        v1 = af.VectorBool(comment="is that so?")
        v2 = af.VectorF32(comment="float like a butterfly")
        v3 = af.VectorI16(comment="int like a three")
    data = aDataset(v1=[True], v2=[1/2], v3=[3])
    test_file = Path("test.parquet")
    data.to_parquet(test_file)
    class KeyValueMetadata(af.Dataset):
        """Stores results of reading Parquet metadata."""
        file_name = af.VectorObject()
        key = af.VectorObject()
        value = af.VectorObject()
    test_file_metadata = KeyValueMetadata.from_sql(
        f"""
        SELECT
            file_name,
            DECODE(key) AS key,
            DECODE(value) AS value,
        FROM parquet_kv_metadata('{test_file}')
        WHERE DECODE(key) != 'ARROW:schema'
        """,
        method="polars",
        field_names="strict"
    )
    test_file.unlink()
    assert all(
        value in test_file_metadata.value.values
        for value in [
            "is that so?", "float like a butterfly", "int like a three",
            "Delightful data.", "manual"
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
        verbatim = IsotopeData.build(query=f"FROM '{url}'")
    data_from_sql = IsotopeData.build(query=f"FROM '{url}'", rename=True)
    assert len(data_from_sql) == 354
    test_file = Path("test.parquet")
    data_from_sql.to_parquet(test_file)
    data_from_parquet = IsotopeData.build(query=f"FROM '{test_file}'")
    test_file.unlink()
    assert data_from_sql == data_from_parquet


def test_nested_dataset():
    class User(af.Dataset):
        name = af.ScalarObject("username")
        attrs = af.VectorObject("user attributes")
    class Order(af.Dataset):
        user = af.VectorObject("user")
        qty = af.VectorI16("quantity")
    u1 = User(name="Alice", attrs=["adorable", "agreeable"])
    u2 = User(name="Brent", attrs=["bland", "broke"])
    o1 = Order(user=[u1, u2], qty=[3, 5])
    assert o1.is_dataset("user") == True
    assert o1.is_dataset("qty") == False