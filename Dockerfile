FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN pip uninstall -y pillow \
    && pip install \
    pillow-simd \
    scipy \
    mlconfig==0.1.8 \
    && rm -rf ~/.cache/pip

WORKDIR /workspace
