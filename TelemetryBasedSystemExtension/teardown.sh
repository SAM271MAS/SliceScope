#!/bin/bash
set -e

# Define your target namespace
NAMESPACE="slicescope"

echo "1. Tearing down Simulation (Users, Env Controller)..."
kubectl delete -f k8s/02-simulation/ -n $NAMESPACE --ignore-not-found=true

echo "2. Tearing down Pipeline (Proxy, Consumer)..."
kubectl delete -f k8s/01-pipeline/ -n $NAMESPACE --ignore-not-found=true

echo "3. Tearing down Infrastructure (Kafka, InfluxDB, Payload Server)..."
kubectl delete -f k8s/00-infrastructure/ -n $NAMESPACE --ignore-not-found=true

echo "Teardown complete!"

