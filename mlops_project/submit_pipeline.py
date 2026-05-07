from kfp import compiler
import kfp

from pipeline import pipeline

PIPELINE_FILE = "pipeline.yaml"


def main():

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=PIPELINE_FILE
    )

    print("Pipeline compiled")

   
    client = kfp.Client(
        host="http://localhost:8080"
    )

   
    client.create_run_from_pipeline_package(
        PIPELINE_FILE,
        arguments={"bucket": "csv-data"}
    )

    print("Pipeline running")


if __name__ == "__main__":
    main()
