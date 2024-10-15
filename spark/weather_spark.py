from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lit, round
spark = SparkSession.builder.appName("Weather Analytics").enableHiveSupport().master("local[*]").getOrCreate()

#Load the Dataset
df = spark.read.csv("weather.csv", header=True, inferSchema=True)

# Display the contents of the Weather Dataset
df.show()

# Data types of the Weather Dataset
df.printSchema()

# Count the no of the rows
df.count()

df_selected = df.select("FullDate", "City", "AvgTemp")
df_selected.show()
df.show()

df_filtered = df.filter((df["MaxTemp"] > 40))
df_filtered.show()

from pyspark.sql import Window
from pyspark.sql.functions import avg, col

window_spec = Window.partitionBy("City").orderBy("Week").rowsBetween(-2, 0)

df_3_week_avg_temp = df.select(
    col("City"), 
    col("Week"), 
    col("AvgTemp"),
    avg("AvgTemp").over(window_spec).alias("3_Week_Avg_Temperature")
)

df_3_week_avg_temp.show()

from pyspark.sql import Window
from pyspark.sql.functions import rank, col


window_spec = Window.partitionBy("City").orderBy(col("AvgTemp").desc())

df_temperature_rank = df.select(
    col("City"), 
    col("AvgTemp"),
    rank().over(window_spec).alias("Temperature_Rank")
)

df_temperature_rank.show()


from pyspark.sql.functions import when, col

df_wind_speed_category = df.select(
    col("WindSpeed"),
    when(col("WindSpeed") < 5, "Low")
    .when(col("WindSpeed") < 15, "Medium")
    .otherwise("High")
    .alias("Wind_Speed_Category")
)

df_wind_speed_category.show()

from pyspark.sql import functions as F

df = df.withColumn(
    "Temperature_Category",
    F.when(F.col("AvgTemp").isNull(), "Unknown")
     .when(F.col("AvgTemp") < 32, "Cold")
     .when(F.col("AvgTemp") < 70, "Mild")
     .otherwise("Hot")
)
df.show()

correlation = df.stat.corr("AvgTemp", "WindSpeed")
correlation

df = df.withColumn("Temperature_Difference", col("MaxTemp") - col("MinTemp"))
df.show()


birmingham_df = df.filter(df.City == "Birmingham")
huntsville_df = df.filter(df.City == "Huntsville")
union_df = birmingham_df.union(huntsville_df)
print("Union of Birmingham and Huntsville data:")
union_df.show()


avg_temp_by_city = df.groupBy("City").agg(round(avg("AvgTemp"), 2).alias("Avg_Temperature"))
avg_temp_by_city.show()

max_min_temp_by_city = df.groupBy("City").agg(
    round(avg("MaxTemp"), 2).alias("Avg_Max_Temperature"),
    round(avg("MinTemp"), 2).alias("Avg_Min_Temperature")
)
max_min_temp_by_city.show()


avg_wind_speed_by_city = df.groupBy("City").agg(round(avg("WindSpeed"), 2).alias("Avg_Wind_Speed"))
avg_wind_speed_by_city.show()

avg_precipitation_by_state = df.groupBy("State").agg(round(avg("Precipitation"), 2).alias("Avg_Precipitation"))
avg_precipitation_by_state.show()


avg_wind_direction_by_city = df.groupBy("City").agg(round(avg("WindDirection"), 2).alias("Avg_Wind_Direction"))
avg_wind_direction_by_city.show()



avg_temp_by_month_city = df.groupBy("Month", "City").agg(round(avg("AvgTemp"), 2).alias("Avg_Temperature"))
avg_temp_by_month_city.show()


total_precipitation_by_year_state = df.groupBy("Year", "State").agg(round(avg("Precipitation"), 2).alias("Total_Precipitation"))
total_precipitation_by_year_state.show()

output_path = "results"
df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

spark.stop()