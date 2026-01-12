FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y \
    python3 python3-pip libgl1 wget git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    torch torchvision \
    insightface \
    opencv-python-headless \
    gfpgan \
    boto3 \
    runpod

ENV INSIGHTFACE_HOME=/models

COPY handler.py .

CMD ["python3", "handler.py"]

