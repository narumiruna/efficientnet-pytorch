FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN pip uninstall -y pillow \
    && pip install \
    pillow-simd \
    scipy \
    mlconfig \
    && rm -rf ~/.cache/pip

WORKDIR /workspace
