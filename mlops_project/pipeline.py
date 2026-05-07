from kfp import dsl, compiler
from build_components import load_op, preprocess_op, train_fnn_op, train_cnn_lstm_op, evaluate_op, train_cnn_op, select_best_model_op,  save_best_model_op

@dsl.pipeline(
    name="mlops-pipeline"
)
def pipeline(bucket: str = "csv-data"):
    download = load_op(bucket=bucket, object_name="data/ue-lte-network-traffic-stats.csv")
    download.set_caching_options(True)

    preprocess_task = preprocess_op(input_csv=download.output)
    preprocess_task.set_caching_options(True)

    train_fnn_task = train_fnn_op(
        input_json=preprocess_task.output
    )

    train_fnn_task.set_caching_options(False)   
    train_cnn_task = train_cnn_op(
        input_json=preprocess_task.output
    )

    train_cnn_task.set_caching_options(False)
    train_cnn_lstm_task = train_cnn_lstm_op(
       input_json=preprocess_task.output
    )

    train_cnn_lstm_task.set_caching_options(False)
    eval_fnn_task = evaluate_op(
       data_input=preprocess_task.output,
       model_input=train_fnn_task.outputs["model_output"]
    )

    eval_fnn_task.set_caching_options(False)
    eval_cnn_task = evaluate_op(
       data_input=preprocess_task.output,
       model_input=train_cnn_task.outputs["model_output"]
    )

    eval_cnn_task.set_caching_options(False)
    eval_cnn_lstm_task = evaluate_op(
       data_input=preprocess_task.output,
       model_input=train_cnn_lstm_task.outputs["model_output"]
    )

    eval_cnn_lstm_task.set_caching_options(False)
    select_best_task = select_best_model_op(
        cnn_metrics=eval_cnn_task.outputs["test_metrics"],  # Utilise les métriques de test
        fnn_metrics=eval_fnn_task.outputs["test_metrics"],
        cnn_lstm_metrics=eval_cnn_lstm_task.outputs["test_metrics"]
    )
    select_best_task.set_caching_options(False)
    save_model_task = save_best_model_op(
        selection_result=select_best_task.outputs["selection_result"],
        cnn_model=train_cnn_task.outputs["model_output"],
        fnn_model=train_fnn_task.outputs["model_output"],
        cnn_lstm_model=train_cnn_lstm_task.outputs["model_output"],
        bucket= "models"
    )
    save_model_task.set_caching_options(False)
   #eval_lstm_task = evaluate_op(
     #  data_input=preprocess_task.output,
      # model_input=train_lstm_task.outputs["model_output"]
    #)

    #eval_lstm_task.set_caching_options(False)
#    monitoring_task = monitoring_op(
 #      input_data=eval_fnn_task.outputs["output_data"]
  #  )
 
