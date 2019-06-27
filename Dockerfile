FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install -U pip \
    && pip install \
    pillow-simd \
    scipy \
    torchvision==0.2.2.post3 \
    && rm -rf ~/.cache/pip

WORKDIR /work
COPY efficientnet .
COPY configs .
COPY train.py .
