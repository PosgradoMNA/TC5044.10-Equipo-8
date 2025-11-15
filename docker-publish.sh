#!/bin/bash

# Docker Hub publishing script
# Usage: ./docker-publish.sh <dockerhub-username> <version>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dockerhub-username> <version>"
    echo "Example: $0 myusername v1.0.0"
    exit 1
fi

USERNAME=$1
VERSION=$2
IMAGE_NAME="energy-efficiency-api"

echo "Building image..."
docker build -t ml-service:latest .

echo "Tagging images..."
docker tag ml-service:latest $USERNAME/$IMAGE_NAME:latest
docker tag ml-service:latest $USERNAME/$IMAGE_NAME:$VERSION

echo "Pushing to Docker Hub..."
docker push $USERNAME/$IMAGE_NAME:latest
docker push $USERNAME/$IMAGE_NAME:$VERSION

echo "Published images:"
echo "  $USERNAME/$IMAGE_NAME:latest"
echo "  $USERNAME/$IMAGE_NAME:$VERSION"
