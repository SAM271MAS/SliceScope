from kfp.components import InputPath, OutputPath

def train_fnn_category(
    input_json: InputPath(str),
    model_output: OutputPath(str),
    metrics_output: OutputPath(str),
    plot_output: OutputPath(str)
):
    import os, json
    import numpy as np
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime
    from sklearn.metrics import r2_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import RootMeanSquaredError
    from tensorflow.keras import backend as K

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    tf.random.set_seed(42)
    np.random.seed(42)

    print("Début de l'entraînement...")

    # --- Load data info ---
    data = np.load(input_json)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    horizon = int(data["horizon"])

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")


    # --- Définir les métriques ---
    def tolerance_accuracy(y_true, y_pred, tol=0.15):
        """
        Calculates the percentage of predictions that fall within a (tol) tolerance
        of the actual values.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        epsilon = K.epsilon()

        absolute_error = tf.abs(y_pred - y_true)
        relative_error = absolute_error / (tf.abs(y_true) + epsilon)

        within_tolerance = tf.less(relative_error, tol)
        accuracy = tf.reduce_mean(tf.cast(within_tolerance, tf.float32))

        return accuracy

    # --- Configuration des epochs ---
    EPOCH_LIMITS = {
        'fnn': 60,
        'cnn': 60,
        'lstm': 60,
        'gru': 60,
        'bi-lstm': 60,
        'cnn-lstm': 60
    }

    def train_optimized_model(model, model_name, X_train, y_train, X_val, y_val, batch_size=64):
        """
        Trains a model using tf.data pipelines, dynamic learning rates, and early stopping.
        Enforces the maximum epochs defined in the reference paper.
        """
        max_epochs = EPOCH_LIMITS.get(model_name.lower(), 100)

        # Calculate a dynamic patience scale (20% of max epochs, minimum of 3)
        patience_scale = max(3, max_epochs // 20)

        # Build High-Speed tf.data Pipelines
        # .cache() keeps data in memory, .prefetch() loads next batch while GPU trains
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Configure Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience_scale,       # Wait dynamically based on model
            restore_best_weights=True,     # Revert to the best epoch automatically
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                    # Cut learning rate in half if plateaued
            patience=patience_scale // 2,  # Trigger LR reduction before Early Stopping kicks in
            min_lr=1e-6,
            verbose=1
        )

        # Train the Model using the pipeline
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=max_epochs,
            callbacks=[early_stop, reduce_lr],
            verbose=2 # Cleaner console output for hundreds of epochs
        )

        return history

    def train_model_wrapper(model, model_name, X_train, y_train, X_val, y_val, epochs, batch_size=64, optimized=True):
        """
        Toggles between standard Keras .fit() and the optimized tf.data pipeline.
        """
        if optimized:
            print(f"--- Training {model_name.upper()} (Optimized Pipeline) ---")
            history = train_optimized_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                batch_size=batch_size
            )
        else:
            print(f"--- Training {model_name.upper()} (Standard .fit) ---")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        return history

    # --- Construction du modèle FNN ---
    def build_fnn_model(input_shape, output_dim):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Flatten())

        # First layer (25 units)
        model.add(Dense(25, activation='relu'))

        # Second layer (25 units)
        model.add(Dense(25, activation='relu'))

        # output layer
        model.add(Dense(output_dim, activation='linear'))

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[
                'mae',
                RootMeanSquaredError(name='rmse'),
                tolerance_accuracy
            ]
        )
        return model

    model = build_fnn_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_dim=y_train.shape[1]
    )

    # Afficher le résumé
    model.summary()

    # --- Entraînement ---

    history_fnn_agg = train_model_wrapper(model, 'fnn', X_train, y_train, X_val, y_val, epochs=60, optimized=True)


    # --- Visualisation ---
    def plot_training_history(history, model_name, plot_output_path):
        """
        Plots the training and validation Loss and Tolerance Accuracy in separate figures
        """
        # Plot Loss (MSE)
        os.makedirs(plot_output_path, exist_ok=True)
        fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))

        ax1.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=1.2)
        ax1.plot(history.history['val_loss'], label='Val Loss', color='red', linestyle='--', linewidth=1.2)

        ax1.set_xlabel('Epochs', fontsize=8)
        ax1.set_ylabel('Loss (MSE)', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        ax1.legend(fontsize=8, frameon=False)
        ax1.grid(True, linestyle=':', alpha=0.5, color='gray')

        plt.tight_layout()
        loss_plot_path = os.path.join(plot_output_path, f'Loss_{model_name.replace("-", "_")}.pdf')
        plt.savefig(loss_plot_path, format='pdf', bbox_inches='tight')
        plt.close()

        # Plot Custom Metric (Tolerance Accuracy)
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))

        if 'tolerance_accuracy' in history.history:
            ax2.plot(history.history['tolerance_accuracy'], label='Train Acc', color='blue', linewidth=1.2)
            ax2.plot(history.history['val_tolerance_accuracy'], label='Val Acc', color='red', linestyle='--', linewidth=1.2)

            ax2.set_xlabel('Epochs', fontsize=8)
            ax2.set_ylabel('Tol. Accuracy', fontsize=8)
            ax2.tick_params(axis='both', labelsize=8)
            ax2.legend(fontsize=8, frameon=False)
            ax2.grid(True, linestyle=':', alpha=0.5, color='gray')

            plt.tight_layout()
            acc_plot_path = os.path.join(plot_output_path, f'Accuracy_{model_name.replace("-", "_")}.pdf')
            plt.savefig(acc_plot_path, format='pdf', bbox_inches='tight')
            plt.close()

    # Sauvegarder les plots
    plots_dir = os.path.join(plot_output, "plots")
    plot_training_history(history_fnn_agg, "FNN_Model_aggregated", plots_dir)


    os.makedirs(model_output, exist_ok=True)

    model_filename = f"fnn_model_h{horizon}.keras"
    model_path = os.path.join(model_output, model_filename)

    model.save(model_path)

    print(f" Modèle sauvegardé: {model_path}")

    # --- Prédictions ---
    y_pred = model.predict(X_val)

    # --- Métriques ---
    metrics = {
        'model_name': 'FNN',
        'horizon': horizon,
        'final_train_loss': float(history_fnn_agg.history['loss'][-1]),
        'final_val_loss': float(history_fnn_agg.history['val_loss'][-1]),
        'best_val_loss': float(min(history_fnn_agg.history['val_loss'])),
        'best_epoch': int(np.argmin(history_fnn_agg.history['val_loss'])),
        'final_val_mae': float(history_fnn_agg.history['val_mae'][-1]),
        'final_val_rmse': float(history_fnn_agg.history['val_rmse'][-1]),
        'final_val_tolerance_acc': float(history_fnn_agg.history['val_tolerance_accuracy'][-1]),
        'r2_score': float(r2_score(y_val.flatten(), y_pred.flatten())),
        'epochs_trained': len(history_fnn_agg.history['loss']),
        'training_completed': datetime.now().isoformat()
    }

    # --- Sauvegarde metrics ---
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n --- Métriques finales ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(" Entraînement terminé avec succès !")

