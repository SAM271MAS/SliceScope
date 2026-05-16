#!/bin/bash

NAMESPACE="slicescope"

echo "=================================================="
echo "1. CHECKING SIMULATION TRAFFIC (USER PODS)"
echo "=================================================="
echo "--- USER 1 (HEAVY) ---"
kubectl logs deploy/user1-heavy --tail=50 -n $NAMESPACE
echo -e "\n--- USER 2 (MEDIUM) ---"
kubectl logs deploy/user2-medium --tail=50 -n $NAMESPACE
echo -e "\n--- USER 3 (LIGHT) ---"
kubectl logs deploy/user3-light --tail=50 -n $NAMESPACE
echo ""

echo "=================================================="
echo "2. CHECKING PROXY & TELEMETRY EXTRACTION"
echo "=================================================="
echo "--- SQUID PROXY ---"
kubectl logs deploy/network-proxy -c squid --tail=50 -n $NAMESPACE
echo -e "\n--- LIVE PRODUCER ---"
kubectl logs deploy/network-proxy -c producer --tail=50 -n $NAMESPACE
echo ""

echo "=================================================="
echo "3. CHECKING KAFKA BROKER"
echo "=================================================="
kubectl logs deploy/kafka-broker --tail=50 -n $NAMESPACE
echo ""

echo "=================================================="
echo "4. CHECKING DATA CONSUMER"
echo "=================================================="
kubectl logs deploy/data-consumer --tail=50 -n $NAMESPACE
echo ""

echo "=================================================="
echo "5. VERIFYING INFLUXDB DATA WRITES"
echo "=================================================="
echo "--- RAW THROUGHPUT KPIs ---"
kubectl -n $NAMESPACE exec -it deploy/influxdb -- influx -database 'network_data' -execute 'SELECT * FROM throughput_kpis ORDER BY time DESC LIMIT 10'
echo -e "\n--- PREDICTED KPIs ---"
kubectl -n $NAMESPACE exec -it deploy/influxdb -- influx -database 'network_data' -execute 'SELECT * FROM predicted_kpis ORDER BY time DESC LIMIT 10'
echo ""

echo "=================================================="
echo "6. TAILING PREDICTOR SERVICE (PRESS CTRL+C TO EXIT)"
echo "=================================================="
# Leaving the -f (follow) flag here so the script hangs and lets you watch the live predictions
kubectl logs deploy/predictor-service --tail=50 -n $NAMESPACE -f
