# Affinity

Affinity makes it easy to create well-annotated datasets from vector data.  
What your data means should always travel together with the data.

Affinity is a pythonic dialect of Data Definition Language (DDL).  Affinity does not replace any dataframe library, but can be used with any package you like.  

## Usage

For now, just copy [`affinity.py`](https://raw.githubusercontent.com/liquidcarbon/affinity/main/affinity.py) into your project.

```python
import affinity as af

class SensorData(af.Dataset):
  """Experimental data from Top Secret Sensor Tech."""
  t = af.VectorF32("elapsed time (sec)")
  channel = af.VectorI8("channel number (left to right)")
  voltage = af.VectorF64("something we measured (mV)")
  is_laser_on = af.VectorBool("are the lights on?")
  exp_id = af.ScalarI32("FK to `experiment`")

# this working concept covers the following:
data = SensorData()                 # ✅ empty dataset
data = SensorData(**fields)         # ✅ build manually
data = SensorData.build(...)        # ✅ build from a source (dataframes, DuckDB)
data.df  # .pl / .arrow             # ✅ view as dataframe (Pandas/Polars/Arrow)
data.metadata                       # ✅ annotations (data dict with column and dataset comments), origin
data.origin                         # ✅ creation metadata, some data provenance
data.to_parquet(...)                # ✅ data.metadata -> Parquet metadata
data.to_csv(...)                    # ⚒️ annotations in the header
data.to_excel(...)                  # ⚒️ annotations on separate sheet
```


## How it works

The `af.Dataset` is Affinity's `BaseModel`, the base class that defines the behavior of children data classes:
- concise class declaration sets the expected dtypes and descriptions for each attribute (column)
- class attributes can be represented by any array (default: `pd.Series` because it handles nullable integers well; available: numpy, polars, arrow)
- class instances can be constructed from any scalars or iterables
- class instances can be cast into any dataframe flavor, and all their methods are available


## Detailed example: Parquet Round-Trip

Affinity makes class declaration as concise as possible.  
All you need to create a data class are typed classes and comments explaining what the fields mean.

#### 1. Declare class

```python
import affinity as af

class IsotopeData(af.Dataset):
    """NIST Atomic Weights & Isotopic Compositions.[^1]
  
    [^1] https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
    """
    symbol = af.VectorObject("Element")
    z = af.VectorI8("Atomic Number (Z)")
    mass = af.VectorF64("Isotope Mass (Da)")
    abundance = af.VectorF64("Relative natural abundance")

IsotopesData.z
# DescriptorType Int8 of len 0  # Atomic Number (Z)
# Series([], dtype: Int8)

IsotopeData().pl  # show fields and types
# shape: (0, 4)
# symbol  z  mass abundance
#    str i8   f64       f64
```

The class attributes are instantiated Vector objects of zero length.  Using the [desciptor pattern](https://docs.python.org/3/howto/descriptor.html), they are replaced with actual data arrays on building the instance.

#### 2. Build class instance from querying a CSV

To build the dataset, we use `IsotopeData.build()` method with `query` argument.  We use DuckDB [FROM-first syntax](https://duckdb.org/docs/sql/query_syntax/from.html#from-first-syntax), with `rename=True` keyword argument.  The fields in the query result will be assigned names and types provided in the class definition.  With `rename=False` (default), the source columns must be named exactly as class attributes.  When safe type casting is not possible, an error will be raised; element with z=128 would not fit this dataset. 

```python
url = "https://raw.githubusercontent.com/liquidcarbon/chembiodata/main/isotopes.csv"
data_from_sql = IsotopeData.build(query=f"FROM '{url}'", rename=True)
# data_from_sql = IsotopeData.build(query=f"FROM '{url}'")  # will fail

query_without_rename = f"""
SELECT
    Symbol as symbol,
    Number as z,
    Mass as mass,
    Abundance as abundance,
FROM '{url}'
"""
data_from_sql2 = IsotopeData.build(query=query_without_rename)
assert data_from_sql == data_from_sql2
print(data_from_sql)

# Dataset IsotopeData of shape (354, 4)
# symbol = ['H', 'H' ... 'Ts', 'Og']
# z = [1, 1 ... 117, 118]
# mass = [1.007825, 2.014102 ... 292.20746, 294.21392]
# abundance = [0.999885, 0.000115 ... 0.0, 0.0]
```

#### 3. Write to Parquet, with metadata.
```python
data_from_sql.to_parquet("test.parquet")  # requires PyArrow
```

#### 4. Inspect metadata using PyArrow:

The schema metadata as shown here is truncated; full-length keys and values are in `pf.schema_arrow.metadata`.

```python
import pyarrow.parquet as pq
pf = pq.ParquetFile("isotopes.parquet")
pf.schema_arrow

# symbol: string
# z: int8
# mass: double
# abundance: double
# -- schema metadata --
# table_comment: 'NIST Atomic Weights & Isotopic Compositions.[^1]
  
#     [' + 97
# symbol: 'Element'
# z: 'Atomic Number (Z)'
# mass: 'Isotope Mass (Da)'
# abundance: 'Relative natural abundance'
# created_ts: '1724787055721'
# source: 'dataframe, shape (354, 4)
# query:

# SELECT
#     Symbol as symbol,
#  ' + 146
```

#### 5. Inspect metadata using DuckDB

DuckDB provides several functions for [querying Parquet metadata](https://duckdb.org/docs/data/parquet/metadata.html).  We're specifically interested in key-value metadata, where both keys and values are of type `BLOB`.  It can be decoded on the fly using `SELECT DECODE(key), DECODE(value) FROM parquet_kv_metadata(...)`, or like so:

```python
import duckdb
source = duckdb.sql("FROM parquet_kv_metadata('isotopes.parquet') WHERE key='source'")
print(source.fetchall()[-1][-1].decode())

# dataframe, shape (354, 4)
# query:

# SELECT
#     Symbol as symbol,
#     Number as z,
#     Mass as mass,
#     Abundance as abundance,
# FROM 'https://raw.githubusercontent.com/liquidcarbon/chembiodata/main/isotopes.csv'
```

#### 6. Read Parquet:

```python
data_from_parquet = IsotopeData.build(query="FROM 'isotopes.parquet'")
assert data_from_sql == data_from_parquet
print(data_from_parquet.pl.dtypes)
# [String, Int8, Float64, Float64]
```



## Motivation

![Tell Me Why](https://github.com/user-attachments/assets/fd985e9b-365e-4d51-96f2-f8165eebfb6c)

Once upon a time, relational databases met object-oriented programming, and gave rise to object-relational mapping. Django ORM and SQLAlchemy unlocked the ability to represent database entries as python objects, with attributes for columns and relations, and methods for create-read-update-delete (CRUD) transactions.  These scalar data elements (numbers, strings, booleans) carry a lot of meaning: someone's name or email or account balance, amounts of available items, time of important events.  They change relatively frequently, one row at a time, and live in active, fast memory (RAM).

`future blurb about OLAP and columnar and cloud data storage`

We need something new for vector data.

There are many options for working with dataframes composed of vectors - pandas, polars, pyarrow are all excellent - there are a few important pieces missing:
1) other than variable and attribute names, there's no good way to explain what the dataset and each field is about; what the data means is separated from the data itself
2) dataframe packages are built for maximum flexibility in working with any data types; this leads to data quality surprises and is sub-optimal for storage and compute

Consider the CREATE TABLE statement in AWS Athena, the equivalent of which does not exist in any one python package:

```sql
CREATE EXTERNAL TABLE [IF NOT EXISTS]
 [db_name.]table_name [(col_name data_type [COMMENT col_comment] [, ...] )]
 [COMMENT table_comment]
 [PARTITIONED BY (col_name data_type [COMMENT col_comment], ...)]
 [CLUSTERED BY (col_name, col_name, ...) INTO num_buckets BUCKETS]
 [ROW FORMAT row_format]
 [STORED AS file_format] 
 [WITH SERDEPROPERTIES (...)]
 [LOCATION 's3://amzn-s3-demo-bucket/[folder]/']
 [TBLPROPERTIES ( ['has_encrypted_data'='true | false',] ['classification'='aws_glue_classification',] property_name=property_value [, ...] ) ]
```

Affinity exists to fill (some of) these gaps.

## Tales of Data Annotation Issues Gone Terribly Wrong

Probably the single greatest source of problems is unspecified units of measure, with numerous fatal and near-fatal engineering and medical disasters.

- [When NASA Lost a Spacecraft Due to a Metric Math Mistake](https://www.simscale.com/blog/nasa-mars-climate-orbiter-metric/)

Have you ever stared at a bunch of numbers and had no clue what they represented?  Do you have an anecdote of bad things happening due un/misannotated data? Share in [discussions](https://github.com/liquidcarbon/affinity/discussions)!

## Future

- pipification?  I envision that even when ~complete, this package remain just one file
- nested data (WIP)

```python
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
```