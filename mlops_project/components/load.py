from kfp.components import OutputPath

def load_minio_data(bucket: str, object_name: str, output: OutputPath(str)):
    from minio import Minio
    client = Minio(
        "minio-service.kubeflow.svc.cluster.local:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )

    client.fget_object(bucket, object_name, output)
    print(f"Fichier {object_name} téléchargé avec succès.")
