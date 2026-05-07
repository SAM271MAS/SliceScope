import os
from minio import Minio
client = Minio(
    "minio-service.kubeflow.svc.cluster.local:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)

bucket = "mlops-models"

# dossier local de téléchargement
download_dir = "minio_downloads"
os.makedirs(download_dir, exist_ok=True)


print(" FICHIERS DANS MINIO:")

objects = client.list_objects(bucket, recursive=True)

for obj in objects:
    object_name = obj.object_name
    local_path = os.path.join(download_dir, object_name)

    # créer dossiers si nécessaire
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"⬇️ Download: {object_name} -> {local_path}")

    client.fget_object(bucket, object_name, local_path)

print(" Download terminé ! Tous les fichiers sont dans:", download_dir)
