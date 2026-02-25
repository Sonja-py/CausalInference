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
    output_df=Output("ri.foundry.main.dataset.7f6a0e74-99b5-4788-b8f4-6898fc009b1e"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
)

def compute(output_df, source_df):
    source_df = source_df.dataframe().toPandas()

    out = {
        "pair": [],
        "count": []
    }

    ingredient_list = source_df.ingredient_concept_id.unique()
    ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)

    for pair in ingredient_pairs:
        dfcopy = source_df.copy()
        df = dfcopy[dfcopy.ingredient_concept_id.isin(list(pair))]
        out["pair"].append(str(pair))
        out["count"].append(str(len(df)))

    results_df = pd.DataFrame(out)

    logger.info(type(results_df))

    # Create a SparkSession object
    spark = SparkSession.builder.getOrCreate()

    # Convert the Pandas DataFrame to a PySpark DataFrame
    df_pyspark = spark.createDataFrame(results_df)

    return output_df.write_dataframe(df_pyspark)
