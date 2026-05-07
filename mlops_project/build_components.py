from kfp.components import create_component_from_func
from components.train_cnn_lstm import train_cnn_lstm_category
from components.load import load_minio_data
from components.preprocess import preprocess
from components.train_cnn import train_cnn_category
from components.train_fnn import train_fnn_category
from components.select_best_model import select_best_model
from components.save_best_model import save_best_model
from components.evaluate import evaluate
BASE_IMAGE = "tibermac/mlops-pipeline:latest"

load_op = create_component_from_func(
    load_minio_data,
    base_image=BASE_IMAGE
)

preprocess_op = create_component_from_func(
    preprocess,
    base_image=BASE_IMAGE
)
train_cnn_lstm_op = create_component_from_func(train_cnn_lstm_category, base_image=BASE_IMAGE)
train_cnn_op = create_component_from_func(train_cnn_category, base_image=BASE_IMAGE)
train_fnn_op = create_component_from_func(train_fnn_category, base_image=BASE_IMAGE)
evaluate_op = create_component_from_func(evaluate, base_image=BASE_IMAGE)
select_best_model_op = create_component_from_func(
    select_best_model,
    base_image=BASE_IMAGE
)
save_best_model_op = create_component_from_func(
    save_best_model,
    base_image=BASE_IMAGE
)
#drift_detection_op = create_component_from_func(drift_detection, base_image=BASE_IMAGE)
#monitoring_op = create_component_from_func(monitoring, base_image=BASE_IMAGE)
