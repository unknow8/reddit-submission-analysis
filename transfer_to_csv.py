# spark-submit transfer_to_csv.py submissions
import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types
import pandas as pd
import os

def main(in_directory):
    df = spark.read.json(in_directory)
    pd_df = df.toPandas()
    os.makedirs('csv', exist_ok=True)
    pd_df.to_csv('csv/submissions.csv', index=False)


if __name__=='__main__':
    in_directory = sys.argv[1]
    spark = SparkSession.builder.appName('Transfer data').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)