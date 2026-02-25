from myproject.TARNet.tarnet import bootstrap_func
from pyspark.sql import SparkSession
from transforms.api import transform, Input, Output, configure

from itertools import combinations
import logging
import pandas as pd

logger = logging.getLogger()


@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_LARGE", "DRIVER_CORES_LARGE"])
@configure(
    [
        "EXECUTOR_GPU_ENABLED",
        "EXECUTOR_MEMORY_MEDIUM",
        "EXECUTOR_CORES_LARGE",
        "NUM_EXECUTORS_8",
    ]
)
@transform(
    output_df=Output("ri.foundry.main.dataset.6621779a-19f8-4dfc-bff9-0fe0b446e3e7"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
    hyperparams_df=Input(
        "ri.foundry.main.dataset.275b5e62-ddd3-435c-a92f-4fe2a9da8c33"
    ),
)
def compute(output_df, source_df, hyperparams_df):
    source_df = source_df.dataframe()
    hyperparams_df = hyperparams_df.dataframe()

    source_df = source_df.toPandas()
    hyperparams_df = hyperparams_df.toPandas()

    results_df = pd.DataFrame()
    # ingredient_pairs = [(40234834, 710062)]
    ingredient_list = source_df.ingredient_concept_id.unique()
    ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)

    mini_batch_size = 10
    mini_batches = []

    for idx in range(0, len(ingredient_pairs), mini_batch_size):
        mini_batches.append(ingredient_pairs[idx : idx + mini_batch_size])

    results_df = bootstrap_func(source_df, hyperparams_df, mini_batches[5])

    logger.info(type(results_df))

    # Create a SparkSession object
    spark = SparkSession.builder.getOrCreate()

    # Convert the Pandas DataFrame to a PySpark DataFrame
    df_pyspark = spark.createDataFrame(results_df)

    return output_df.write_dataframe(df_pyspark)
