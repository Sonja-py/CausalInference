from myproject.TARNet.tarnet import tarnet_func
from pyspark.sql import SparkSession
from transforms.api import transform, Input, Output, configure

# from multiprocessing import Pool
from itertools import combinations
import logging
import pandas as pd

logger = logging.getLogger()


@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_LARGE", "DRIVER_CORES_LARGE"])
@configure(
    [
        "EXECUTOR_GPU_ENABLED",
        "EXECUTOR_MEMORY_MEDIUM",
        "EXECUTOR_CORES_SMALL",
        "EXECUTOR_CORES_EXTRA_SMALL",
    ]
)
@transform(
    output_df=Output("ri.foundry.main.dataset.9a9925f9-7dd5-48d1-894c-aaac3f68fa62"),
    output_counterfactual_df=Output("ri.foundry.main.dataset.e995eb2f-e112-4b98-b992-767fce182fdc"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
)
def compute(output_df, output_counterfactual_df, source_df):
    source_df = source_df.dataframe()

    source_df = source_df.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = source_df.ingredient_concept_id.unique()
    ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)

    mini_batch_size = 10
    mini_batches = []

    for idx in range(0, len(ingredient_pairs), mini_batch_size):
        mini_batches.append(ingredient_pairs[idx : idx + mini_batch_size])

    results_df, counterfactual_df = tarnet_func(source_df, mini_batches[1])
    logger.info(type(results_df))

    # Create a SparkSession object
    spark = SparkSession.builder.getOrCreate()

    # Convert the Pandas DataFrame to a PySpark DataFrame
    df_pyspark = spark.createDataFrame(results_df)
    # counter_pyspark = spark.createDataFrame(counterfactual_df)
    # output_counterfactual_df.write_dataframe(counter_pyspark)

    return output_df.write_dataframe(df_pyspark)
