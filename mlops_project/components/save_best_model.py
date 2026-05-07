from kfp.components import InputPath, OutputPath

def save_best_model(
    selection_result:  InputPath(str),
    cnn_model:         InputPath(str),
    fnn_model:         InputPath(str),
    cnn_lstm_model:    InputPath(str),         
    model_registry:    OutputPath(str),
    bucket: str = "models"
):
    import json
    import os
    import tempfile
    from minio import Minio
    from datetime import datetime

    print("="*80)
    print("SAUVEGARDE DU MEILLEUR MODÈLE DANS MINIO")
    print("="*80)

    # -----------------------------
    # LOAD SELECTION
    # -----------------------------
    with open(selection_result, 'r') as f:
        selection = json.load(f)

    best_model = selection['best_model']

    print(f"\n Meilleur modèle: {best_model}")
    print(f"   {selection['reasoning']}")
    print(f"   R² moyen: {selection['avg_r2'][best_model]:.4f}")
    print(
        f"   Victoires R²: "
        f"CNN={selection['feature_wins']['CNN']}, "
        f"FNN={selection['feature_wins']['FNN']}, "
        f"CNN-LSTM={selection['feature_wins']['CNN-LSTM']}"    
    )

    # -----------------------------
    # MINIO
    # -----------------------------
    client = Minio(
        "minio-service.kubeflow.svc.cluster.local:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"\n Bucket créé: {bucket}")

    # -----------------------------
    # SELECT MODEL                
    # -----------------------------
    if best_model == "CNN-LSTM":
        model_dir     = cnn_lstm_model
        model_type    = "CNN-LSTM"
        features_perf = selection['cnn_lstm_features']
    elif best_model == "CNN":
        model_dir     = cnn_model
        model_type    = "CNN"
        features_perf = selection['cnn_features']
    else:  # FNN
        model_dir     = fnn_model
        model_type    = "FNN"
        features_perf = selection['fnn_features']

    # -----------------------------
    # FIND MODEL FILE
    # -----------------------------
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]

    if not model_files:
        raise ValueError(f"Aucun modèle trouvé dans {model_dir}")

    local_model_path = os.path.join(model_dir, model_files[0])

    # -----------------------------
    # NAMING
    # -----------------------------
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    avg_r2     = selection['avg_r2'][best_model]
    r2_percent = int(avg_r2 * 100)

    safe_type        = model_type.replace("-", "_")             
    model_filename   = f"{safe_type}_model_r2{r2_percent}_{timestamp}.keras"
    metrics_filename = f"{safe_type}_metrics_r2{r2_percent}_{timestamp}.json"

    # -----------------------------
    # UPLOAD MODEL
    # -----------------------------
    remote_model_path = f"best_models/{model_filename}"
    client.fput_object(bucket, remote_model_path, local_model_path)
    print(f"\n Modèle uploadé: {bucket}/{remote_model_path}")

    # -----------------------------
    # SAVE METRICS
    # -----------------------------
    metrics_to_save = {
        'best_model':        best_model,
        'selection_criteria': 'R2',
        'saved_at':          datetime.now().isoformat(),
        'avg_r2':            avg_r2,
        'feature_wins':      selection['feature_wins'],
        'total_features':    selection['total_features'],
        'features_performance': {
            feature: {
                'r2':       features_perf[feature]['r2'],
                'accuracy': features_perf[feature].get('accuracy', 0),
                'rmse':     features_perf[feature].get('rmse', 0),
                'mae':      features_perf[feature].get('mae', 0)
            }
            for feature in features_perf
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(metrics_to_save, tmp, indent=2)
        tmp_metrics_path = tmp.name

    remote_metrics_path = f"best_models/{metrics_filename}"
    client.fput_object(bucket, remote_metrics_path, tmp_metrics_path)
    os.unlink(tmp_metrics_path)
    print(f" Métriques uploadées: {bucket}/{remote_metrics_path}")

    # -----------------------------
    # LATEST COPY
    # -----------------------------
    latest_model   = f"latest/{safe_type}_latest.keras"        
    latest_metrics = f"latest/{safe_type}_latest.json"

    client.fput_object(bucket, latest_model, local_model_path)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(metrics_to_save, tmp, indent=2)
        tmp_metrics_path = tmp.name

    client.fput_object(bucket, latest_metrics, tmp_metrics_path)
    os.unlink(tmp_metrics_path)
    print(f" Copie latest: {bucket}/{latest_model}")

    # -----------------------------
    # REGISTRY
    # -----------------------------
    registry_info = {
        'best_model':       best_model,
        'saved_at':         datetime.now().isoformat(),
        'model_location':   f"s3://{bucket}/{remote_model_path}",
        'metrics_location': f"s3://{bucket}/{remote_metrics_path}",
        'avg_r2':           avg_r2,
        'feature_wins':     selection['feature_wins'],
        'total_features':   selection['total_features']
    }

    with open(model_registry, 'w') as f:
        json.dump(registry_info, f, indent=2)

    print("\n" + "="*80)
    print(" SAUVEGARDE TERMINÉE")
    print(f"   Modèle:    {model_filename}")
    print(f"   R² moyen:  {avg_r2:.4f} ({r2_percent}%)")
    print(f"   Victoires: {selection['feature_wins'][best_model]}/{selection['total_features']}")
    print(f"   Emplacement: s3://{bucket}/{remote_model_path}")
    print("="*80)
