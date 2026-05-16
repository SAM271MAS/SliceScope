#!/bin/bash
set -e

NAMESPACE="slicescope"

echo "1. Deploying Infrastructure (Kafka, InfluxDB, Payload Server)..."
kubectl apply -f k8s/00-infrastructure/ -n $NAMESPACE

echo "Waiting for infrastructure to initialize"
kubectl wait --for=condition=available deployment/payload-server deployment/kafka-broker deployment/influxdb -n $NAMESPACE --timeout=5m

echo "2. Deploying Pipeline (Proxy, Consumer)..."
kubectl apply -f k8s/01-pipeline/ -n $NAMESPACE

echo "3. Deploying Simulation (Env Controller, Users)..."
kubectl apply -f k8s/02-simulation/ -n $NAMESPACE

echo "Deployment complete! Check your pods with: kubectl get pods -n $NAMESPACE"

