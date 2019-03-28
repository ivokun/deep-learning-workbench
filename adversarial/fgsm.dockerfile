FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install git -y

WORKDIR /app
RUN pip install git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans
