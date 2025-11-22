# Docker for Development with NVIDIA GPU

This guide explains how to use Docker to create a consistent development environment for this project, especially when using NVIDIA GPUs for tasks like profiling.

## Docker for Development

This `Dockerfile` is designed to create a consistent and reproducible development environment for this project. It includes all the necessary dependencies, including the Rust toolchain, Python packages, and NVIDIA CUDA support for GPU profiling.

### Building the Docker Image

To build the Docker image, run the following command from the root of the project:

```bash
docker build -t cs336-dev .
```

This will create a Docker image named `cs336-dev` with all the dependencies and source code.

### Running the Docker Container

To run the Docker container for development, you need to mount the local source code into the container. This allows you to edit the code on your local machine and have the changes reflected inside the container immediately.

You also need to expose the GPUs to the container to enable profiling and other GPU-accelerated tasks.

Use the following command to run the container:

```bash
docker run -it --gpus all -v $(pwd):/app cs336-dev /bin/bash
```

This command does the following:

-   `docker run`: Starts a new Docker container.
-   `-it`: Runs the container in interactive mode with a terminal.
-   `--gpus all`: Makes all available GPUs accessible to the container.
-   `-v $(pwd):/app`: Mounts the current directory (the project root) into the `/app` directory inside the container.
-   `cs336-dev`: The name of the image to use.
-   `/bin/bash`: Starts a bash shell inside the container, which you can use for development.

Once you are inside the container's shell, you can run all the project commands, such as `uv run pytest`, as you would in a local environment.

### Cloud Compute (Brev.dev)

When using a cloud compute provider like [Brev.dev](https://brev.dev), you can use the same Docker container to ensure a consistent environment. Brev.dev provides GPU-enabled instances, and you can use the Docker extension in VS Code to build and run the container.
