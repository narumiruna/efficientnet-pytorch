import argparse

import numpy as np
import tensorflow as tf
import torch

import efficientnet
from eval_ckpt_main import EvalCkptDriver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='efficientnet-b0')
    parser.add_argument('--image-file', type=str, default='panda.jpg')
    return parser.parse_args()


def get_trained_parameters(model_name, image_files, ckpt_dir):
    driver = EvalCkptDriver(model_name=model_name)

    trained_params = []
    with tf.Graph().as_default(), tf.Session() as sess:
        images, _ = driver.build_dataset(image_files, [0], False)
        driver.build_model(images, is_training=False)
        driver.restore_model(sess, ckpt_dir)

        for v in tf.global_variables():
            name = v.name
            param = sess.run(v)

            if 'depthwise_conv2d/depthwise_kernel' in name:
                param = np.transpose(param, (2, 3, 0, 1))
            elif 'conv2d' in name and 'kernel' in name:
                param = np.transpose(param, (3, 2, 0, 1))
            elif 'dense' in name and 'kernel' in name:
                param = np.transpose(param, (1, 0))

            trained_params.append(torch.from_numpy(param))

    return trained_params


def main():
    args = parse_args()
    trained_params = get_trained_parameters(args.model_name, [args.image_file], args.model_name)

    model = getattr(efficientnet, args.model_name.replace('-', '_'))()
    model.eval()

    state_dict = model.state_dict()

    for k, v in state_dict.items():
        if 'num_batches_tracked' in k:
            continue
        state_dict[k] = trained_params.pop(0)

    model.load_state_dict(state_dict)
    torch.save(state_dict, f'{args.model_name}.pth')

if __name__ == '__main__':
    main()
