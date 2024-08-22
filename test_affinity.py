import affinity as af
import numpy as np
import pandas as pd
import pytest

def test_empty_vector_no_dtype():
    with pytest.raises(TypeError):
        v = af.Vector()


def test_empty_vector():
    v = af.Vector(np.int8)
    assert len(v) == 0
    assert repr(v) == "Vector <class 'numpy.int8'> of len 0  # None\narray([], dtype=int8)"

def test_typed_vectors():
    v_untyped = af.VectorUntyped()
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

def test_empty_dataset():
    class aDataset(af.Dataset):
        v = af.Vector(np.int8)
    data = aDataset()
    assert data.df.shape == (0, 1)
    assert data.df.dtypes["v"] == np.int8


def test_simple_dataset():
    class aDataset(af.Dataset):
        v1 = af.Vector(np.float32, comment="first")
        v2 = af.Scalar(np.int8, comment="second")
    data = aDataset(
        v1 = [0., 1.],
        v2 = 2
    )
    assert len(data) == 2
    assert data.data_dict == {"v1": "first", "v2": "second"}
    expected_df = pd.DataFrame({
        "v1": [0., 1.],
        "v2": [2, 2]
    }).astype({"v1": np.float32, "v2": np.int8})
    pd.testing.assert_frame_equal(data.df, expected_df)
