# Affinity

Affinity makes it easy to create well-annotated datasets from vector data.  
What your data means should always travel together with the data.

```
import affinity as af

class SensorData(af.Dataset):
  """Experimental data from Top Secret Sensor Tech."""
  exp_id: af.ScalarI32("FK to `experiment`")
  t: af.VectorF32("elapsed time")
  channel: af.VectorI8("channel number")
  is_laser_on: af.VectorBool("are the lights on?")
  voltage: af.VectorF64("something we measured")

data = SensorData()                 # ✅ empty dataset
data = SensorData(**fields)         # ✅ build manually
data = SensorData.build(...)        # ⚒️ build from external source, validate types
data.df                             # ✅ view as dataframe (Pandas/Polars/Arrow)
data.sql(query)                     # ⚒️ query data (DuckDB)
data.data_dict                      # ✅ column and table comments
data.to_csv(...)                    # ⚒️ annotations in the header
data.to_excel(...)                  # ⚒️ annotations on separate sheet
data.to_parquet(...)                # ⚒️ data types, annotations in Parquet metadata
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

## Future

- nested data
- pipification?  I envision that even when ~complete, this package remain just one file
