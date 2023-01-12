# Base image
FROM python:3.7-slim

# Image from anibali to use GPU
#FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04 


# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy relevant files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

COPY models/ models/
COPY reports/ reports/

# set workdir
WORKDIR /

# install packages
RUN pip install -r requirements.txt --no-cache-dir

# Entrypoint (running application when the image is executed)
ENTRYPOINT ["python", "-u", "src/models/predict_model.py models/mnist/trained_model.pt data/raw/test_img_mnist"]