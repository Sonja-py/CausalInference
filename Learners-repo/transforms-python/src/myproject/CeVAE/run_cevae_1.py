from pyspark.sql import SparkSession
from transforms.api import transform, Input, Output, configure
import pandas as pd
from itertools import combinations
import logging
from myproject.CeVAE.cevae_main_no_fold import cevae_func

logger = logging.getLogger()


@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_EXTRA_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_EXTRA_LARGE", "EXECUTOR_CORES_LARGE"])
@transform(
    output_df=Output("ri.foundry.main.dataset.5be41d98-9869-4dea-8533-729364278b33"),
    debug_output=Output("ri.foundry.main.dataset.dd790a04-08a9-4041-96d8-034c21d5ca73"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
)
def compute(output_df, debug_output, source_df):
    top10 = [(717607, 725131), (750982, 725131), (44507700, 725131),
    (715939, 725131), (755695, 725131), (778268, 722031), (797617, 725131),
    (739138, 725131), (717607, 715259), (738156, 725131)]

    source_df = source_df.dataframe()
    source_df = source_df.toPandas()

    results_df = pd.DataFrame()
    ingredient_list = source_df.ingredient_concept_id.unique()
    ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)
    mini_batches = []
    
    for i, pair in enumerate(ingredient_pairs):
        if pair in top10[:3]:
            mini_batches.append(pair)
    # try:
    results_df = cevae_func(source_df, mini_batches)
    # except Exception as e:
    #     df = pd.DataFrame.from_dict({"log": e})
    #     logger.info(type(results_df))
    #     spark = SparkSession.builder.getOrCreate()
    #     df_pyspark = spark.createDataFrame(df)
    #     return debug_output.write_dataframe(df_pyspark)


    # results_df = tarnet(source_df)
    logger.info(type(results_df))

    # Create a SparkSession object
    spark = SparkSession.builder.getOrCreate()

    # Convert the Pandas DataFrame to a PySpark DataFrame
    df_pyspark = spark.createDataFrame(results_df)

    return output_df.write_dataframe(df_pyspark)
