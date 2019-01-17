from __future__ import print_function
from __future__ import division
import argparse
from glob import glob

import tensorflow as tf
 
from model_UNET_plus import saliency
from utils import *
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for sgd')
parser.add_argument('--cost', dest='cost', default='xepy', help='cost to minimize')
parser.add_argument('--aug_scale', dest='aug_scale', type=int, default=2, help='scale of data augmentation')

parser.add_argument('--train_set', dest='train_set', default='Imgs_256', help='training data path')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='tensorboard logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='MSRA-B', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='ECSSD', help='dataset for testing')

parser.add_argument('--gpu', dest='gpu', default='0', help='which gpu to use')
parser.add_argument('--type', dest='type', default='', help='arg to give unique names to realizations')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')

args = parser.parse_args()

#training function
def saliency_train(saliency, lr):
    
    #train set filepaths
    train_imgs = sorted(glob('./data/{}/*.jpg'.format(args.train_set)))
    train_maps = sorted(glob('./data/{}/*.png'.format(args.train_set)))
    
    #validation set filepaths
    validation_imgs = sorted(glob('./data/{}/*.jpg'.format(args.eval_set)))
    validation_maps = sorted(glob('./data/{}/*.png'.format(args.eval_set)))
    
    #training function
    saliency.train(train_imgs, train_maps, validation_imgs[:200], validation_maps[:200], batch_size=args.batch_size, epoch=args.epoch, lr=lr)


def saliency_test(saliency, save_dir):
    print('Testing on {} dataset'.format(args.test_set))
    
    #test set filepaths
    test_imgs = sorted(glob('./data/{}/*.jpg'.format(args.test_set)))
    test_maps = sorted(glob('./data/{}/*.png'.format(args.test_set)))
    
    #test function
    saliency.test(test_imgs, test_maps, save_dir)

def main(_):
    
    #the following string is attached to checkpoint, log and image folder names
    name = "UNET_plus_" + args.cost + "_aug" + str(args.aug_scale) + str(args.type)
    
    ckpt_dir = args.ckpt_dir + "/" + name
    sample_dir = args.sample_dir + "/" + name
    test_dir = args.test_dir + "/" + name
    log_dir = args.log_dir + "/" + name
    
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    #learning rate decay schedule
    lr = args.lr * np.ones([args.epoch])
    lr[int(0.8*args.epoch):] = lr[0] / 10.0 #lr decay
    #lr[80:] = lr[0] / 10.0
    
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = saliency(sess, args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir, aug_scale = args.aug_scale, folder_names=name)
            if args.phase == 'train':
                saliency_train(model, lr=lr)
            elif args.phase == 'test':
                saliency_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = saliency(sess, args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir, aug_scale = args.aug_scale, folder_names=name)
            if args.phase == 'train':
                saliency_train(model, lr=lr)
            elif args.phase == 'test':
                saliency_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    tf.app.run()
