import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek
from pyspark.sql.types import *

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = config.get('AWS', 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Description: Initiate the Spark session to be used by the pipeline
    :return: spark
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Description: Read in the song files from the S3 bucket named in 'input_data_ and write out 2 new tables;
        songs_table
        artists_table.
    Both are written back in parquet format to the output_data directory.
    :param spark: The spark session substantiated in create_spark_session()
    :param input_data: The location of the data to be loaded (S3 bucket in this case)
    :param output_data: The location for the parquet tables to be written to
    :return: N/A
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*"
    
    # read song data file
    df = spark.read.json(song_data).drop_duplicates()

    # extract columns to create songs table
    songs_table = df.select("song_id",
                            "title",
                            "artist_id",
                            "year",
                            "duration"
                            ).drop_duplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + "songs_table/",
                              mode="overwrite",
                              partitionBy=["year", "artist_id"]
                              )

    # extract columns to create artists table
    artists_table = df.select("artist_id",
                              "artist_name",
                              "artist_location",
                              "artist_latitude",
                              "artist_longitude"
                              ).drop_duplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists_table/",
                                mode="overwrite"
                                )


def process_log_data(spark, input_data, output_data):
    """
    Description: Read in the log files from the S3 bucket named in 'input_data_ and write out 3 new tables;
        users_table
        time_table
        songplays_table.
    All are written back in parquet format to the output_data directory.
    :param spark: The spark session substantiated in create_spark_session()
    :param input_data: The location of the data to be loaded (S3 bucket in this case)
    :param output_data: The location for the parquet tables to be written to
    :return: N/A
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*.json"

    # read log data file
    log_df = spark.read.json(log_data).drop_duplicates()
    
    # filter by actions for song plays
    log_df = log_df.filter(log_df.page == "NextSong")

    # extract columns for users table    
    users_table = log_df.select("userId",
                                "firstName",
                                "lastName",
                                "gender",
                                "level",
                                ).drop_duplicates()
    
    # write users table to parquet files
    users_table.write.parquet(output_data + "users_table/",
                              mode="overwrite"
                              )

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x / 1000, TimestampType())
    log_df = log_df.withColumn("time_stamp", get_timestamp("ts"))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), TimestampType())
    log_df = log_df.withColumn("date_time", get_datetime(log_df.time_stamp))
    
    # extract columns to create time table
    time_table = log_df.withColumn("hour", hour("date_time")) \
        .withColumn("day", dayofmonth("date_time")) \
        .withColumn("week", weekofyear("date_time")) \
        .withColumn("month", month("date_time")) \
        .withColumn("year", year("date_time")) \
        .withColumn("weekday", dayofweek("date_time")) \
        .select("ts", "date_time", "hour", "day", "week", "month", "year", "weekday").drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + "time_table/",
                             mode='overwrite',
                             partitionBy=["year", "month"]
                             )

    # read in song data to use for songplays table
    song_df = spark.read.option("basePath", output_data + "songs_table/")\
        .format("parquet")\
        .load(output_data + "songs_table/*/*/*")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = log_df.join(song_df, log_df.song == song_df.title, how='inner')\
        .select(monotonically_increasing_id().alias("songplay_id"),
                col("date_time"),
                col("userId").alias("user_id"),
                col("level"),
                col("song_id"),
                col("artist_id"),
                col("sessionId").alias("session_id"),
                col("location"),
                col("userAgent").alias("user_agent")
                )

    songplays_table = songplays_table.join(time_table, songplays_table.date_time == time_table.date_time,
                                           how="inner") \
        .select(col("songplay_id"),
                songplays_table.date_time,
                col("user_id"),
                col("level"),
                col("song_id"),
                col("artist_id"),
                col("session_id"),
                col("location"),
                col("user_agent"),
                col("year"),
                col("month")
                )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.drop_duplicates().write.parquet((output_data + "songplays_table/"),
                                                    mode="overwrite",
                                                    partitionBy=["year", "month"]
                                                    )


def main():
    """

    :return:
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    # input_data = "/home/sam/Documents/Udacity Data Engineer/Project 4 - Data Lake/Udacity---Data-Lake-Project/"
    output_data = "/home/sam/Documents/Udacity Data Engineer/Project 4 - Data Lake/Udacity---Data-Lake-Project/Output/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
