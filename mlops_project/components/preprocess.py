from kfp.components import InputPath, OutputPath

def preprocess(
    input_csv: InputPath(str),
    output: OutputPath(str)
):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import tempfile
    import re

    print("Début du prétraitement...")

    ts_df = pd.read_csv(input_csv)

    window_size = 30
    horizon = 5

    KPIS = ["web-rtc", "sipp", "web-server", "Jitter", "CQI"]

    features = [
        col for col in ts_df.columns
        if any(kpi in col for kpi in KPIS)
    ]

    def sort_features(features):
        def extract(col):
            user_match = re.search(r"UE(\d+)", col)
            user_id = int(user_match.group(1)) if user_match else 9999
            kpi_index = next((i for i, k in enumerate(KPIS) if k in col), 999)
            return (user_id, kpi_index)

        return sorted(features, key=extract)

    features = sort_features(features)

    if len(features) == 0:
        raise ValueError("Aucune feature valide trouvée")

    # --- SPLIT ---
    total_rows = len(ts_df)
    train_cutoff = int(total_rows * 0.7)
    val_cutoff = train_cutoff + int(total_rows * 0.15)

    train_df = ts_df.iloc[:train_cutoff]
    val_df = ts_df.iloc[train_cutoff:val_cutoff]
    test_df = ts_df.iloc[val_cutoff:]

    # --- SCALER ---
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df[features].fillna(0))
    val_scaled = scaler.transform(val_df[features].fillna(0))
    test_scaled = scaler.transform(test_df[features].fillna(0))

    # --- CONTEXT ---
    val_scaled = np.vstack((train_scaled[-window_size:], val_scaled))
    test_scaled = np.vstack((val_scaled[-window_size:], test_scaled))

    def create_sliding_window(data, window_size=30, horizon=1):
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:i + window_size])
            future = data[i + window_size:i + window_size + horizon]
            y.append(np.mean(future, axis=0))
        return np.array(X), np.array(y)

    X_train, y_train = create_sliding_window(train_scaled, window_size, horizon)
    X_val, y_val = create_sliding_window(val_scaled, window_size, horizon)
    X_test, y_test = create_sliding_window(test_scaled, window_size, horizon)

    # --- SCALER SAVE ---
    tmp_scaler = tempfile.mktemp(suffix=".save")
    joblib.dump(scaler, tmp_scaler)

    with open(tmp_scaler, "rb") as f:
        scaler_bytes = f.read()

    # --- SAVE ALL ---
    with open(output, "wb") as f:
        np.savez_compressed(
            f,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            horizon=horizon,
            scaler=scaler_bytes,
            feature_names=np.array(features)
        )

    print("Prétraitement terminé")

