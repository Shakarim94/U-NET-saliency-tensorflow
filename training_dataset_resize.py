from __future__ import print_function
from __future__ import division

import os
import numpy as np
from PIL import Image
from utils import *
from glob import glob


def func():
    data_path = './data/Imgs'
    save_path = './data/Imgs_256'
    
    maps_list = sorted(glob('{}/*.png'.format(data_path)))
    imgs_list = sorted(glob('{}/*.jpg'.format(data_path)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    for idx in xrange(len(imgs_list)):
        im = Image.open(imgs_list[idx]).convert('RGB')
        im = im.resize((256,256), resample=Image.BICUBIC)
        im.save('{}/{}.jpg'.format(save_path, idx))


        map = Image.open(maps_list[idx]).convert('L')
        map = map.resize((256,256), resample=Image.BICUBIC)
        map.save('{}/{}.png'.format(save_path, idx))

if __name__ == '__main__':
    func()
