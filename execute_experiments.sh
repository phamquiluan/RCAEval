#!/bin/bash
set -e  # Exit on any command failure

# Check if Python version argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_version>"
    echo "Supported versions: python3.8, python3.12"
    exit 1
fi

PYTHON_VERSION=$1

# Set base image based on Python version
case $PYTHON_VERSION in
    "python3.8")
        BASE_IMAGE="ghcr.io/astral-sh/uv:python3.8-bookworm-slim"
        INSTALLATION_TAG="rcd"
        ;;
    "python3.12")
        BASE_IMAGE="ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
        INSTALLATION_TAG="default"
        ;;
    *)
        echo "Unsupported Python version: $PYTHON_VERSION"
        echo "Supported versions: python3.8, python3.12"
        exit 1
        ;;
esac

echo "Building Docker image with Python version: $PYTHON_VERSION"
echo "Base image: $BASE_IMAGE"
echo "Installation tag: $INSTALLATION_TAG"

# Create nested directories
mkdir -p $(pwd)/ctn_data/data_raw
find $(pwd)/ctn_data/data_raw -type d -exec chmod 777 {} +
mkdir -p $(pwd)/ctn_data/data_prepared
find $(pwd)/ctn_data/data_prepared -type d -exec chmod 777 {} +
mkdir -p $(pwd)/ctn_data/output
find $(pwd)/ctn_data/output -type d -exec chmod 777 {} +

docker build --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg INSTALLATION_TAG=$INSTALLATION_TAG -t rca-eval-experiments-${PYTHON_VERSION} .
docker run -it --rm \
    -v $(pwd)/ctn_data/data_raw:/app/data_raw \
    -v $(pwd)/ctn_data/data_prepared:/app/data \
    -v $(pwd)/ctn_data/output:/app/output \
    --cpus=8 \
    --memory=8g \
    --memory-swap=8g \
    rca-eval-experiments-${PYTHON_VERSION}
