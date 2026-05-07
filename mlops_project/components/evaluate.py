from kfp.components import InputPath, OutputPath

def evaluate(
    data_input: InputPath(str),
    model_input: InputPath(str),
    output_data: OutputPath(str),
    plot_output: OutputPath(str),
    test_metrics: OutputPath(str)
):
    import os
    import json
    import numpy as np
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
    import tempfile

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("=== Début évaluation ===")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    data = np.load(data_input, allow_pickle=True)

    X_test  = data["X_test"]
    y_test  = data["y_test"]
    horizon = int(data["horizon"])
    FEATURES = data["feature_names"]

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model_files = os.listdir(model_input)
    model_path  = os.path.join(model_input, model_files[0])
    model       = tf.keras.models.load_model(model_path, compile=False)

    # -----------------------------
    # DETECT MODEL NAME           
    # -----------------------------
    def detect_model_name(path: str) -> str:
        p = path.lower()
        if "cnn-lstm" in p:
            return "CNN-LSTM"
        elif "cnn" in p:
            return "CNN"
        elif "fnn" in p:
            return "FNN"

    model_name = detect_model_name(model_path)
    print(f"Modèle détecté : {model_name}")

    # -----------------------------
    # PREDICTION
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # LOAD SCALER
    # -----------------------------
    scaler_bytes = data["scaler"].item()

    tmp_scaler = tempfile.mktemp(suffix=".save")
    with open(tmp_scaler, "wb") as f:
        f.write(scaler_bytes)

    scaler = joblib.load(tmp_scaler)
    os.remove(tmp_scaler)

    # -----------------------------
    # INVERSE TRANSFORM
    # -----------------------------
    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(y_pred)

    # -----------------------------
    # FEATURE TYPES
    # -----------------------------
    feature_types = {}
    for f in FEATURES:
        if "Jitter" in f:
            feature_types[f] = "jitter"
        elif "CQI" in f:
            feature_types[f] = "cqi"
        else:
            feature_types[f] = "throughput"

    # -----------------------------
    # METRICS FUNCTIONS
    # -----------------------------
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return {
            "rmse": float(rmse),
            "mae":  float(mae),
            "r2":   float(r2),
        }

    # -----------------------------
    # METRICS PER FEATURE
    # -----------------------------
    all_metrics = {}

    print("\n" + "=" * 80)
    print("MÉTRIQUES PAR FEATURE")
    print("=" * 80)
    print(f"{'Feature':<20} {'RMSE':<10} {'MAE':<10} {'R2':<8}")
    print("-" * 70)

    for i, feature in enumerate(FEATURES):
        if i >= y_test_unscaled.shape[1]:
            break

        y_true_f = y_test_unscaled[:, i]
        y_pred_f = y_pred_unscaled[:, i]

        metrics = calculate_metrics(y_true_f, y_pred_f)
        all_metrics[feature] = metrics

        print(
            f"{feature:<20} "
            f"{metrics['rmse']:<10.4f} "
            f"{metrics['mae']:<10.4f} "
            f"{metrics['r2']:<8.4f}"
        )

    # -----------------------------
    # SAVE METRICS
    # -----------------------------
    with open(test_metrics, "w") as f:
        json.dump({
            "model_name": model_name,
            "horizon":    horizon,
            "features":   all_metrics
        }, f, indent=2)

    # -----------------------------
    # PLOT FUNCTION
    # -----------------------------
    def plot_feature(actual, predicted, feature_name, metrics, horizon):
        fig, ax = plt.subplots(figsize=(10, 4))

        steps = min(200, len(actual))
        t     = range(steps)

        ax.plot(t, actual[:steps],    label="Actual",    linewidth=1.5)
        ax.plot(t, predicted[:steps], label="Predicted", linestyle="--", linewidth=1.5)

        if "web" in feature_name or "sipp" in feature_name:
            ylabel = "Throughput (Kbps)"
        elif "Jitter" in feature_name:
            ylabel = "Jitter (ms)"
        elif "CQI" in feature_name:
            ylabel = "CQI"
        else:
            ylabel = "Value"

        ax.set_xlabel("Time Slots")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"[{model_name}] {feature_name}\n"   
            f"MAE={metrics['mae']:.3f}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe = feature_name.replace(":", "").replace(" ", "_")
        plt.savefig(os.path.join(plot_output, f"{model_name}_h{horizon}_{safe}.pdf"))  
        plt.close()

    # -----------------------------
    # CREATE PLOTS
    # -----------------------------
    os.makedirs(plot_output, exist_ok=True)

    for i, feature in enumerate(FEATURES):
        if i < y_test_unscaled.shape[1]:
            plot_feature(
                y_test_unscaled[:, i],
                y_pred_unscaled[:, i],
                feature,
                all_metrics[feature],
                horizon
            )

    # -----------------------------
    # SAVE PREDICTIONS
    # -----------------------------
    with open(output_data, "wb") as f:
        np.savez_compressed(
            f,
            y_pred=y_pred_unscaled,
            y_true=y_test_unscaled,
            feature_names=FEATURES
        )

    print(f"\n✔ {len(FEATURES)} features évaluées")
    print("=== Fin évaluation ===")
