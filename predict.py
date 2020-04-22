import myfunc as mf

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sb
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile, Image
from torchvision import datasets, transforms, models

def args_paser():
    pa = argparse.ArgumentParser(description='predictor')
    pa.add_argument('--checkpoint', type=str, default='checkpoint_cmd.pth', help='checkpoint to str')
    pa.add_argument('--cuda', type=bool, default='True', help='True: cuda on gpu, False: cpu')
    pa.add_argument('--topk', type=int, default=5, help='top classes')
    pa.add_argument('--img', type=str, required='True',help='Path of image')

    args = pa.parse_args()
    return args

def main():
    args = args_paser()

    device = mf.set_device(use_gpu = args.cuda)

    model = mf.load_checkpoint(args.checkpoint)
    model = model.to(device)
    classes, names = mf.map_idx_to_class(model.class_to_idx)
    probs, preds = mf.predict(args.img, model, classes, device, args.topk)

    print(args.img)
    print('Predicted top classes : ', preds)
    print('Flowers: ', [names[i] for i in preds])
    print('Probablity: ', probs)

    image_plot = False
    if image_plot == True:
        image = mf.process_image(args.img)
        mf.imshow(image)
        probs, preds = mf.predict(args.img, model, classes, device, args.topk)
        names = [names[i] for i in preds]
        # Display topk most probable flower categories
        _, ax = plt.subplots()
        y_pos = np.arange(len(names))
        ax.barh(y_pos, probs, align='center', color='blue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # labels read top-to-bottom
        plt.title(image_path)
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
            sys.exit(0)
