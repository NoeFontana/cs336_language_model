# Docker for Development with NVIDIA GPU

## Docker for Development

This `Dockerfile` is designed to create a consistent and reproducible development environment for this project. When using a cloud compute provider, you can use the same Docker image to ensure a consistent environment.
The `Dockerfile` includes all the necessary dependencies, including the Rust toolchain, Python packages, and NVIDIA CUDA support for GPU profiling.

### Building and Publishing the Docker Image

Login to ghcr.io and run

```bash
make docker-release DOCKER_USERNAME=your_lowercase_username DOCKER_TAG=tag_version
```

### Running the Docker Container

To run the Docker container for development, you may want to mount the local source code into the container. This allows you to edit the code on your local machine and have the changes reflected inside the container immediately.

Use the following command to run the container:

```bash
docker run -it --gpus all -v $(pwd):/app cs336-dev /bin/bash
```
