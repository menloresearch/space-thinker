docker build -t robot-reasoning-api .
docker run --gpus all -p 8000:8000 -d --name robot-reasoning-service robot-reasoning-api