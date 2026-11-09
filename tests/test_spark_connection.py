from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("AirOps-Connection-Test")
    .master("spark://localhost:7077")
    .getOrCreate()
)

print("Spark Version:", spark.version)

df = spark.range(1, 11)

print("Total records:", df.count())

df.show()

spark.stop()