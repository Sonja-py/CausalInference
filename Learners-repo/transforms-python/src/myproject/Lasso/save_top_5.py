# from pyspark.sql import functions as F
from transforms.api import transform, Input, Output, configure
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType


import pandas as pd

@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_EXTRA_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_LARGE"])
@transform(
    output = Output("ri.foundry.main.dataset.58516625-13eb-4b88-90f1-21cecc1ce7bb"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def compute(source_df, output):
    source_df = source_df.dataframe()
    source_df = source_df.toPandas()
    combination = (717607, 725131)
    df = source_df.copy()
    df = df[df.ingredient_concept_id.isin(list(combination))]

    # Assign treatment label
    df["treatment"] = df["ingredient_concept_id"].apply(
        lambda x: 0 if x == combination[0] else 1
    )

    # Separate based on outcome
    df_y1 = df[df["severity_final"] == 1]
    df_y0 = df[df["severity_final"] == 0]

    # Target size is 1/3 of original data
    target_total = int(len(df) / 3)
    target_y1_frac = 0.3
    target_y1 = int(target_total * target_y1_frac)
    target_y0 = target_total - target_y1

    # Sample with desired balance
    df_y1_sampled = df_y1.sample(n=min(target_y1, len(df_y1)), random_state=42)
    df_y0_sampled = df_y0.sample(n=target_y0, random_state=42)

    df_sampled = pd.concat([df_y1_sampled, df_y0_sampled])
    df_sampled = df_sampled.sample(frac=1.0, random_state=42)  # shuffle

    # Select only specified features
    features = [
        "SICKLECELLDISEASE_before_or_day_of_covid_indicator",
        "DEMENTIA_before_or_day_of_covid_indicator",
        "METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator",
        "PSYCHOSIS_before_or_day_of_covid_indicator",
        "HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator",
    ]
    
    final_df = df_sampled[features + ["treatment", "severity_final"]]

    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Define the schema (all binary ints)
    schema = StructType([
        StructField("SICKLECELLDISEASE_before_or_day_of_covid_indicator", IntegerType(), True),
        StructField("DEMENTIA_before_or_day_of_covid_indicator", IntegerType(), True),
        StructField("METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator", IntegerType(), True),
        StructField("PSYCHOSIS_before_or_day_of_covid_indicator", IntegerType(), True),
        StructField("HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator", IntegerType(), True),
        StructField("treatment", IntegerType(), True),
        StructField("severity_final", IntegerType(), True),
    ])

    # Convert to PySpark DataFrame with schema
    df_pyspark = spark.createDataFrame(final_df, schema=schema)

    # Output the PySpark DataFrame
    return output.write_dataframe(df_pyspark)