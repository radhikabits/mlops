#!/bin/bash

# Pull latest image
docker pull $IMAGE

# Stop and remove existing container
docker rm -f mlops_api || true

# Start new container
docker run -d \
  --name mlops_api \
  -p 8000:8000 \
  --env-file ./api/.env.api \
  $IMAGE
