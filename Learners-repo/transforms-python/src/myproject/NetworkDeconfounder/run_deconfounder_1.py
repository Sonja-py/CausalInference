from pyspark.sql import SparkSession
from transforms.api import transform, Input, Output, configure
import pandas as pd
from itertools import combinations
import logging
from myproject.NetworkDeconfounder.network_creation import compute_network_deconfounder

logger = logging.getLogger()


@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_LARGE"])
@transform(
    output_df=Output("ri.foundry.main.dataset.8bc277b1-a0b3-4737-b922-070399fa3c8a"),
    concept_set_members=Input(
        "ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"
    ),
    final_data=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
    condition_mappings=Input(
        "ri.foundry.main.dataset.b5c3702d-9383-4057-83ac-39a1844d007d"
    ),
    temp_person_condition_network=Input(
        "ri.foundry.main.dataset.de088514-7d31-45ae-9898-35a88c46bc27"
    ),
    drug_condition=Input(
        "ri.foundry.main.dataset.d6a29338-ca1a-4f13-85fd-7b82498484b4"
    ),
    person_drug=Input("ri.foundry.main.dataset.7b33023a-1c07-4ae6-88d3-e9d83de7ab7a"),
    drug_drug=Input("ri.foundry.main.dataset.9eaab353-758b-4e66-a6a8-0ccb22935b0c"),
    condition_condition=Input(
        "ri.foundry.main.dataset.f3c801ad-fc9f-4356-95f0-491480362527"
    ),
    person_condition=Input(
        "ri.foundry.main.dataset.de088514-7d31-45ae-9898-35a88c46bc27"
    ),
    condition_concept=Input(
        "ri.foundry.main.dataset.0678aacb-6b1f-45ba-894e-5fe02a196f54"
    ),
)
def compute(
    output_df,
    temp_person_condition_network,
    concept_set_members,
    final_data,
    condition_mappings,
    drug_condition,
    person_drug,
    drug_drug,
    condition_condition,
    person_condition,
    condition_concept,
):
    concept_set_members = concept_set_members.dataframe()
    final_data = final_data.dataframe()
    drug_condition_network = drug_condition.dataframe()
    person_drug_network = person_drug.dataframe()
    drug_drug_network = drug_drug.dataframe()
    condition_condition_network = condition_condition.dataframe()
    person_condition_network = person_condition.dataframe()
    condition_concept_network = condition_concept.dataframe()

    results_df = pd.DataFrame()

    ingredient_list = final_data.toPandas().ingredient_concept_id.unique()
    ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)

    mini_batch_size = 10
    mini_batches = []

    for idx in range(0, len(ingredient_pairs), mini_batch_size):
        mini_batches.append(ingredient_pairs[idx : idx + mini_batch_size])

    results_df = compute_network_deconfounder(
        final_data,
        mini_batches[0],
        person_condition_network,
        person_drug_network,
        drug_condition_network,
        drug_drug_network,
        condition_condition_network,
        condition_concept_network,
    )

    # Create a SparkSession object
    spark = SparkSession.builder.getOrCreate()

    # Convert the Pandas DataFrame to a PySpark DataFrame
    df_pyspark = spark.createDataFrame(results_df)

    return output_df.write_dataframe(df_pyspark)
