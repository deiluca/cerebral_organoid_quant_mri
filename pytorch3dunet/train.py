# MIT License

# Copyright (c) 2018 Adrian Wolny (https://github.com/wolny/pytorch-3dunet)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import random

import torch
import sys
sys.path.insert(0, '/home/ws/oc9627/cerebral_organoid_quant_mri')  # noqa

from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning(
            'Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
    # create trainer
    trainer = create_trainer(config)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
