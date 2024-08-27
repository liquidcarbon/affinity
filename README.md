# Affinity

Affinity makes it easy to create well-annotated datasets from vector data.  
What your data means should always travel together with the data.

## Usage

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
data = SensorData.build(...)        # ✅ build from a source (dataframes, DuckDB) with type casting
data.df  # .pl / .arrow             # ✅ view as dataframe (Pandas/Polars/Arrow)
data.metadata                       # ✅ annotations (data dict with column and dataset comments), origin
data.origin                         # ✅ creation metadata, some data provenance
data.to_parquet(...)                # ✅ data.metadata -> Parquet metadata
data.to_csv(...)                    # ⚒️ annotations in the header
data.to_excel(...)                  # ⚒️ annotations on separate sheet
```

## Parquet Round-Trip

1) declare class
2) build class instance from a URL using DuckDB [FROM-first syntax](https://duckdb.org/docs/sql/query_syntax/from.html#from-first-syntax), renaming and retyping the fields on the fly
3) write to Parquet (with metadata)

```python
import affinity as af

class IsotopeData(af.Dataset):
    """NIST Atomic Weights & Isotopic Compositions.[^1]
  
    [^1] https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
    """
    symbol = af.VectorUntyped("Element")
    z = af.VectorI8("Atomic Number (Z)")
    mass = af.VectorF64("Isotope Mass (Da)")
    abundance = af.VectorF64("Relative natural abundance")

url = "https://raw.githubusercontent.com/liquidcarbon/chembiodata/main/isotopes.csv"
query_without_rename = f"""
SELECT
    Symbol as symbol,
    Number as z,
    Mass as mass,
    Abundance as abundance,
FROM '{url}'
"""
data_from_sql = IsotopeData.build(query="FROM '{url}'", rename=True)
data_from_sql.to_parquet("test.parquet")
```

4) inspect metadata using PyArrow:

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

5) inspect metadata using DuckDB

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

6) read Parquet:

```python
data_from_parquet = IsotopeData.build(query="FROM 'isotopes.parquet'")
assert data_from_sql == data_from_parquet
print(data_from_parquet.pl.dtypes)
# [String, Int8, Float64, Float64]
```

## How it works

Affinity does not replace any dataframe library, but can be used with any package you like.  

The `af.Dataset` is a base class that defines the behavior of children data classes:
- the concise class definition carries the annotations and expected data types
- subclass attributes (vectors) can be represented by any array (numpy, pandas, polars, arrow)
- subclass instances can be constructed from any scalars or iterables
- subclass instances can be cast into any dataframe flavor, and exported to any format that your favorite dataframes support

## Motivation

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

- nested data
- pipification?  I envision that even when ~complete, this package remain just one file
