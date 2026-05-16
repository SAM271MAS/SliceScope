#!/bin/bash

NAMESPACE="slicescope"

echo "Routing Grafana Dashboard..."

kubectl port-forward svc/grafana-service 3000:3000 -n $NAMESPACE > /dev/null 2>&1 &

GRAFANA_PID=$!

echo "Grafana is now live!"
echo "   -> Open your local browser to: http://localhost:3000"
echo "   -> To stop the port-forward later, run: kill $GRAFANA_PID"

