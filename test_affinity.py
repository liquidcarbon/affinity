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
