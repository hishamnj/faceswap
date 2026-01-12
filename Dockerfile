FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ca-certificates \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
    torch torchvision \
    insightface \
    opencv-python==4.8.1.78 \
    gfpgan \
    boto3 \
    fastapi \
    uvicorn

ENV INSIGHTFACE_HOME=/models
ENV PYTHONUNBUFFERED=1

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
