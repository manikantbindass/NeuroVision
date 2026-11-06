# NeuroVision Helm chart

One release occupies a dedicated namespace. The chart creates API, worker, frontend, schema initialization, service, configuration and network-policy resources. API autoscaling and TLS ingress are opt-in.

Provision PostgreSQL and Redis first, then supply the existing `neurovision-runtime` secret with `database-url`, `redis-url` and `api-token` keys. Values contain only secret references. The schema hook must be able to create the initial tables.

Use built image tags or immutable digests. The default local tags are for a local cluster with images already loaded. The worker uses the same application image with a separate command. GPU mode needs a CUDA-capable worker image, NVIDIA device plugin, schedulable GPU nodes, and appropriate node selection.

See [deployment instructions](../../../docs/deployment/README.md) for local, Kubernetes and cloud operation.
