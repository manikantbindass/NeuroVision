# Local CLI container

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

Run from the repository root with Docker Engine in Linux container mode and the Docker Compose plugin:

```text
docker compose config --quiet
docker compose run --build --rm pipeline
```

The service prints the pipeline stages and exits. The runtime uses a non-root user, a read-only root filesystem, and no network access. The build needs network access to retrieve the base image and Python build tools. Kubernetes, Helm, and Terraform directories are reserved for future services.
