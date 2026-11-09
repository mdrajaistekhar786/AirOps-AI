from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("AirOps-Parquet-Verification")
    .master("local[*]")
    .getOrCreate()
)

df = spark.read.parquet("storage/processed/flights")

print("Total flight records:", df.count())

df.printSchema()

df.show(5, truncate=False)

spark.stop()