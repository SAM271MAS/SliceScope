#!/bin/bash
                                                                                                                      
set -e                                   

BASE="$(pwd)/src"

echo "Importing images into Kubernetes containerd..."
ctr -n k8s.io images import "$BASE/traffic_generator/traffic-gen.tar"
ctr -n k8s.io images import "$BASE/data_producer/live-producer.tar"
ctr -n k8s.io images import "$BASE/data_consumer/data-consumer.tar"
ctr -n k8s.io images import "$BASE/env_controller/env-controller.tar"
ctr -n k8s.io images import "$BASE/predictor/predictor.tar"

echo "All images successfully imported."

