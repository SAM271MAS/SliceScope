#!/bin/bash      

set -e

BASE="$(pwd)/src"

echo "Building traffic generator..."
cd "$BASE/traffic_generator"
docker build -t traffic-gen:v1 .
docker save traffic-gen:v1 -o traffic-gen.tar

echo "Building live producer..."
cd "$BASE/data_producer"
docker build -t live-producer:v1 .
docker save live-producer:v1 -o live-producer.tar

echo "Building data consumer..."
cd "$BASE/data_consumer"
docker build -t data-consumer:v1 .
docker save data-consumer:v1 -o data-consumer.tar

echo "Building Environment Controller..."
cd "$BASE/env_controller"
docker build -t env-controller:v1 .
docker save env-controller:v1 -o env-controller.tar

echo "Building Predictor Service..."
cd "$BASE/predictor"
docker build -t predictor:v1 .
docker save predictor:v1 -o predictor.tar

echo "All images built and saved successfully as v1."
