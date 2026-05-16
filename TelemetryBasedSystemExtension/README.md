# Deployment & Execution Instructions

### 1. Build and Import Images

First, compile the core microservice container images and load them directly into the cluster's runtime:
> **Note:** The `import_images.sh` script assumes you are using `ctr` (raw containerd).

```bash
./build_images.sh
./import_images.sh
```

### 2. Deploy the Environment

Execute the setup script to deploy the Environment

```bash
./deploy_images.sh
```

### 3. Access the Dashboards

Once the deployment finishes and the pods are running, route the Grafana dashboard to your local machine:

```bash
./start_dashboards.sh
```

Access the UI at http://localhost:3000. To visualize the telemetry, configure an InfluxDB datasource (Name: InfluxDB, URL: http://influxdb-service:9084, Database: network_data) and import the provided dashboard.json template.

### 4. Verify Execution

Run the verification script:

```bash
./verify_environment.sh
```

### 5. Teardown

Remove all pipeline and infrastructure resources from the cluster:

```bash
./teardown.sh

```

# Technical Design Notes & Validation Scope

This system extension was built as a functional framework to validate data pipeline mechanics and execute performance tests. To prioritize rapid logic verification over production overhead, the following design choices were deliberately made:
- **Ephemeral Storage:** Persistent storage layers (PVCs) were omitted.
- **Environment Fallbacks:** Microservice variables rely on the `os.getenv("VAR", "fallback")` pattern to maximize portability during local execution.
- **Dependency Management:** Strict package version pinning was (mostly) omitted from this phase in favor of pulling stable, base environment setups.

