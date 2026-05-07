from kfp.components import InputPath, OutputPath

def select_best_model(
    cnn_metrics:      InputPath(str),
    fnn_metrics:      InputPath(str),
    cnn_lstm_metrics: InputPath(str),       
    selection_result: OutputPath(str)
):
    import json
    import numpy as np

    print("="*80)
    print("SÉLECTION DU MEILLEUR MODÈLE - R²")
    print("="*80)

    # -----------------------------
    # LOAD METRICS
    # -----------------------------
    with open(cnn_metrics, 'r') as f:
        cnn = json.load(f)
    with open(fnn_metrics, 'r') as f:
        fnn = json.load(f)
    with open(cnn_lstm_metrics, 'r') as f:          
        cnn_lstm = json.load(f)

    features = list(cnn['features'].keys())
    print(f"\nComparaison sur {len(features)} features\n")

    # -----------------------------
    # STATS
    # -----------------------------
    feature_wins = {'CNN': 0, 'FNN': 0, 'CNN-LSTM': 0}       
    feature_comparison = {}

    print(f"{'Feature':<20} {'CNN R²':<12} {'FNN R²':<12} {'CNN-LSTM R²':<14} {'Diff':<10} {'Winner'}")
    print("-"*90)

    for f in features:
        cnn_r2      = cnn['features'][f]['r2']
        fnn_r2      = fnn['features'][f]['r2']
        cnn_lstm_r2 = cnn_lstm['features'][f]['r2']            

        best_r2 = max(cnn_r2, fnn_r2, cnn_lstm_r2)           

        if best_r2 == cnn_lstm_r2 and cnn_lstm_r2 > max(cnn_r2, fnn_r2):
            winner = "CNN-LSTM"
            feature_wins['CNN-LSTM'] += 1
        elif cnn_r2 >= fnn_r2 and cnn_r2 >= cnn_lstm_r2:
            if cnn_r2 == fnn_r2 == cnn_lstm_r2:
                winner = "EQUAL"
            else:
                winner = "CNN"
                feature_wins['CNN'] += 1
        elif fnn_r2 >= cnn_r2 and fnn_r2 >= cnn_lstm_r2:
            winner = "FNN"
            feature_wins['FNN'] += 1
        else:
            winner = "CNN-LSTM"
            feature_wins['CNN-LSTM'] += 1

        diff = cnn_lstm_r2 - max(cnn_r2, fnn_r2)               

        feature_comparison[f] = {
            "cnn_r2":      cnn_r2,
            "fnn_r2":      fnn_r2,
            "cnn_lstm_r2": cnn_lstm_r2,
            "diff":        diff,
            "winner":      winner
        }

        print(
            f"{f:<20} {cnn_r2:<12.4f} {fnn_r2:<12.4f} "
            f"{cnn_lstm_r2:<14.4f} {diff:<+10.4f} {winner}"
        )

    # -----------------------------
    # AVERAGE R2
    # -----------------------------
    avg_cnn      = np.mean([cnn['features'][f]['r2']      for f in features])
    avg_fnn      = np.mean([fnn['features'][f]['r2']      for f in features])
    avg_cnn_lstm = np.mean([cnn_lstm['features'][f]['r2'] for f in features])  

    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    print(f"CNN R² moyen:      {avg_cnn:.4f}")
    print(f"FNN R² moyen:      {avg_fnn:.4f}")
    print(f"CNN-LSTM R² moyen: {avg_cnn_lstm:.4f}")            

    # -----------------------------
    # DECISION
    # -----------------------------
    avg_scores = {'CNN': avg_cnn, 'FNN': avg_fnn, 'CNN-LSTM': avg_cnn_lstm}   
    best_model = max(avg_scores, key=avg_scores.get)
    best_avg   = avg_scores[best_model]

    # Vérifier s'il y a égalité sur le R² moyen
    tied = [m for m, v in avg_scores.items() if v == best_avg]

    if len(tied) == 1:
        reasoning = f"{best_model} a meilleur R² moyen ({best_avg:.4f})"
    else:
        # Départager par nombre de features gagnées
        best_model = max(tied, key=lambda m: feature_wins[m])
        reasoning  = f"égalité moyenne mais {best_model} gagne plus de features"

    print("\n" + "="*80)
    print(f"MEILLEUR MODÈLE: {best_model}")
    print(reasoning)
    print("="*80)

    # -----------------------------
    # SAVE RESULT
    # -----------------------------
    results = {
        "best_model":    best_model,
        "reasoning":     reasoning,
        "avg_r2": {
            "CNN":      float(avg_cnn),
            "FNN":      float(avg_fnn),
            "CNN-LSTM": float(avg_cnn_lstm)                     
        },
        "feature_wins":    feature_wins,
        "total_features":  len(features),
        "feature_comparison": feature_comparison,               
        "cnn_features":      cnn["features"],
        "fnn_features":      fnn["features"],
        "cnn_lstm_features": cnn_lstm["features"]              
    }

    with open(selection_result, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✔ Résultat sauvegardé")
