from pyspark.sql import DataFrame
from transforms.api import transform, Input, Output, configure
from functools import reduce


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
    output_df=Output("ri.foundry.main.dataset.0d645ea3-8041-482e-a548-ea708421e06b"),
    ds1=Input("ri.foundry.main.dataset.a8fa3bc7-d453-4bcc-946c-47a4e0aad9fb"),
    ds2=Input("ri.foundry.main.dataset.9a9925f9-7dd5-48d1-894c-aaac3f68fa62"),
    ds3=Input("ri.foundry.main.dataset.b907dc36-269c-4e95-8f4b-f323e74bc10d"),
    ds4=Input("ri.foundry.main.dataset.d1cf32b4-3cfe-4ec9-aca6-740a12b0e8b2"),
    ds5=Input("ri.foundry.main.dataset.11111eb6-024b-4f41-b72d-a897eb245533"),
    ds6=Input("ri.foundry.main.dataset.f1b70f65-19bf-4108-98a7-506beb957fbe"),
    ds7=Input("ri.foundry.main.dataset.99409918-10d2-437e-8da3-d2939eed3215"),
    ds8=Input("ri.foundry.main.dataset.9181fa3b-dc08-4377-8c39-7bedffed6337"),
    ds9=Input("ri.foundry.main.dataset.d5f39f62-9227-48bc-abab-116dd81d0f88"),
    ds10=Input("ri.foundry.main.dataset.2330f61a-cce3-4046-9ba1-715c4d341350"),
    ds11=Input("ri.foundry.main.dataset.6c4161cc-8896-49ea-8792-6d2cbe552ce4"),
    ds12=Input("ri.foundry.main.dataset.3ecd86ea-ef5d-4944-b77d-e78ac10cd33d"),
    ds13=Input("ri.foundry.main.dataset.215d2abd-b58b-417a-8793-6724b4771e8b"),
    ds14=Input("ri.foundry.main.dataset.375c6929-1442-442e-be81-c2320eab897d"),
    ds15=Input("ri.foundry.main.dataset.8ebecd8d-89d4-4386-9e19-66c66b042f67"),
    ds16=Input("ri.foundry.main.dataset.02565d66-6582-40ef-b528-c2f3d2f4925f"),
)
# ri.foundry.main.dataset.02565d66-6582-40ef-b528-c2f3d2f4925f
# ri.foundry.main.dataset.0d645ea3-8041-482e-a548-ea708421e06b
def compute(
    output_df,
    ds1,
    ds2,
    ds3,
    ds4,
    ds5,
    ds6,
    ds7,
    ds8,
    ds9,
    ds10,
    ds11,
    ds12,
    ds13,
    ds14,
    ds15,
    ds16,
):
    ds1 = ds1.dataframe()
    ds2 = ds2.dataframe()
    ds3 = ds3.dataframe()
    ds4 = ds4.dataframe()
    ds5 = ds5.dataframe()
    ds6 = ds6.dataframe()
    ds7 = ds7.dataframe()
    ds8 = ds8.dataframe()
    ds9 = ds9.dataframe()
    ds10 = ds10.dataframe()
    ds11 = ds11.dataframe()
    ds12 = ds12.dataframe()
    ds13 = ds13.dataframe()
    ds14 = ds14.dataframe()
    ds15 = ds15.dataframe()
    ds16 = ds16.dataframe()

    dfs = [
        ds1,
        ds2,
        ds3,
        ds4,
        ds5,
        ds6,
        ds7,
        ds8,
        ds9,
        ds10,
        ds11,
        ds12,
        ds13,
        ds14,
        ds15,
        ds16,
    ]
    df = reduce(DataFrame.unionAll, dfs)

    # Create a SparkSession object
    # spark = SparkSession.builder.getOrCreate()

    # # Convert the Pandas DataFrame to a PySpark DataFrame
    # df_pyspark = spark.createDataFrame(results_df)

    return output_df.write_dataframe(df)
