from pyspark.sql import SparkSession


def create_spark_session():
    return (
        SparkSession.builder
        .appName("AirOpsAI")
        .master("spark://spark-master:7077")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1"
        )
        .config("spark.sql.shuffle.partitions", "3")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )