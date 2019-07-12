#!/bin/bash
wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
for i in {0..5};
do
    wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-b${i}.tar.gz
    tar zxf efficientnet-b${i}.tar.gz
    python3 convert.py --model-name efficientnet-b${i}
done
