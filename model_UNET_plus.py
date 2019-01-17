#Computer Vision Final Project

#importing 
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import time
import os
from utils import *

#this the plain UNET network
def UNET(input, is_training=True, output_channels=1):
    he_init = tf.contrib.layers.variance_scaling_initializer() #initializer for the conv layers
    num_filts = 64 #num of filters in the first convolution
    with tf.variable_scope('Encoder1'):
        conv1 = tf.layers.batch_normalization(tf.layers.conv2d(input, num_filts, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(conv1, num_filts, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='Max_pooling1')
    with tf.variable_scope('Encoder2'):
        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(max_pool1, num_filts * 2, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv4 = tf.layers.batch_normalization(tf.layers.conv2d(conv3, num_filts * 2, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2, name='Max_pooling2')
    with tf.variable_scope('Encoder3'):
        conv5 = tf.layers.batch_normalization(tf.layers.conv2d(max_pool2, num_filts * 4, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv6 = tf.layers.batch_normalization(tf.layers.conv2d(conv5, num_filts * 4, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        max_pool3 = tf.layers.max_pooling2d(conv6, pool_size=[2, 2], strides=2, name='Max_pooling3')
    with tf.variable_scope('Encoder4'):
        conv7 = tf.layers.batch_normalization(tf.layers.conv2d(max_pool3, num_filts * 8, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv8 = tf.layers.batch_normalization(tf.layers.conv2d(conv7, num_filts * 8, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        max_pool4 = tf.layers.max_pooling2d(conv8, pool_size=[2, 2], strides=2, name='Max_pooling4')
    with tf.variable_scope('Bootleneck'):
        conv9 = tf.layers.batch_normalization(tf.layers.conv2d(max_pool4, num_filts * 16, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv10 = tf.layers.batch_normalization(tf.layers.conv2d(conv9, num_filts * 16, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv11 = tf.layers.batch_normalization(tf.layers.conv2d(conv10, num_filts * 16, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
    with tf.variable_scope('Decoder4'):
        size4_h = 2 * tf.shape(conv11)[1]
        size4_w = 2 * tf.shape(conv11)[2]
		#here we apply nearest-neighbor 2x2 upsampling
        upsample4 = tf.image.resize_images(conv11, size=[size4_h, size4_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat4 = tf.concat([conv8, upsample4], 3) #concatenating with one of the encoder layer outputs
        conv12 = tf.layers.batch_normalization(tf.layers.conv2d(concat4, num_filts * 8, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv13 = tf.layers.batch_normalization(tf.layers.conv2d(conv12, num_filts * 8, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
    with tf.variable_scope('Decoder3'):
        size3_h = 2 * tf.shape(conv13)[1]
        size3_w = 2 * tf.shape(conv13)[2]
        upsample3 = tf.image.resize_images(conv13, size=[size3_h, size3_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat3 = tf.concat([conv6, upsample3], 3)
        conv14 = tf.layers.batch_normalization(tf.layers.conv2d(concat3, num_filts * 4, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv15 = tf.layers.batch_normalization(tf.layers.conv2d(conv14, num_filts * 4, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
    with tf.variable_scope('Decoder2'):
        size2_h = 2 * tf.shape(conv15)[1]
        size2_w = 2 * tf.shape(conv15)[2]
        upsample2 = tf.image.resize_images(conv15, size=[size2_h, size2_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat2 = tf.concat([conv4, upsample2], 3)
        conv16 = tf.layers.batch_normalization(tf.layers.conv2d(concat2, num_filts * 2, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv17 = tf.layers.batch_normalization(tf.layers.conv2d(conv16, num_filts * 2, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
    with tf.variable_scope('Decoder1'):
        size1_h = 2 * tf.shape(conv17)[1]
        size1_w = 2 * tf.shape(conv17)[2]
        upsample1 = tf.image.resize_images(conv17, size=[size1_h, size1_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat1 = tf.concat([conv2, upsample1], 3)
        conv18 = tf.layers.batch_normalization(tf.layers.conv2d(concat1, num_filts, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        conv19 = tf.layers.batch_normalization(tf.layers.conv2d(conv18, num_filts, 3, padding='same', kernel_initializer=he_init, activation=tf.nn.relu, use_bias=False), training=is_training)
        output = tf.layers.conv2d(conv18, filters=output_channels, kernel_size=1, padding='same') #1x1 convolution
    return output #[batch_size, height, width, 1]


class saliency(object):

    def __init__(self, sess, cost_str, ckpt_dir, sample_dir, log_dir, aug_scale, folder_names):
        self.sess = sess
        self.name = folder_names
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        self.aug_scale = aug_scale
        
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
		
		#placeholders for the input image, its saliency map and batchnorm training status
        self.img_tensor = tf.placeholder(tf.float32, [None, None, None, 3], name='input_image')
        self.map_tensor = tf.placeholder(tf.float32, [None, None, None, 1], name='binary_map')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
		
		#we apply sigmoid layer to the output of UNET
        with tf.variable_scope('CV'):
            self.pred_tensor = UNET(self.img_tensor, is_training=self.is_training)
            self.pred_tensor_sigmoid = tf.nn.sigmoid(self.pred_tensor)
			
        #COST FUNCTIONS (mse and binary cross entropy)
        self.mse = tf.losses.mean_squared_error(self.map_tensor, self.pred_tensor_sigmoid) 
        self.xepy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred_tensor, labels=self.map_tensor))
		
		#here one of the two cost functions are chosen to be minimized 
        if cost_str == 'mse':
            self.cost = self.mse
        elif cost_str == 'xepy':
            self.cost = self.xepy       
        
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf.placeholder(tf.float32, name='eva_psnr') #avg. psnr of the validation set
        self.eva_mae = tf.placeholder(tf.float32, name='eva_mae') #avg. mae of the validation set

		#Adam optimizer with default settings
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

		#these lines are for batchnorm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.cost)
			
        self.saver = tf.train.Saver(max_to_keep=5) #checpoint saver
		
        init = tf.global_variables_initializer() #initializing variables
        self.sess.run(init)
        print('[*] Initialize model successfully...')
        return

	#this function evaluates the performance on validation set
    def evaluate(self, test_imgs, test_maps, iter_num, summ_eval, summary_writer):
        num_test = len(test_imgs)
        mae_sum = 0
        psnr_sum = 0
        for idx in xrange(num_test):
			#loading the image and map
            image_256, image_orig, size = load_images_RGB_float32(test_imgs[idx])
            map_256, map_orig = load_images_float32(test_maps[idx])
			
			#passing through network to get the saliency map prediction
            predicted_map_256= self.sess.run(self.pred_tensor_sigmoid, feed_dict={self.img_tensor: image_256, self.map_tensor: map_256, self.is_training: False})
            
			#resizing the saliency map from (256,256) back to original size
            predicted_map_256 = Image.fromarray(np.clip(255 * np.squeeze(predicted_map_256), 0, 255).astype('uint8'))
            predicted_map = predicted_map_256.resize(size, resample=Image.BICUBIC)
            predicted_map = np.expand_dims(np.expand_dims(np.array(predicted_map), axis=0), axis=3)
            
			#converting to uint8
            inputimage = np.clip(255 * image_orig, 0, 255).astype('uint8')
            saliencymap = np.clip(255 * map_orig, 0, 255).astype('uint8')
            outputmap = predicted_map
            
			#metrics
            psnr = cal_psnr(saliencymap, predicted_map.astype(np.float32))
            mae = cal_mae(map_orig, predicted_map.astype(np.float32) / 255.0)
            mae_sum += mae
            psnr_sum += psnr
			
			#saving the 20 images for viewing
            if idx<20:
                print('img%d MAE: %.6f' % (idx + 1, mae))
                save_images(os.path.join(self.sample_dir, 'eval%d_%d_concat.png' % (idx + 1, iter_num)), inputimage, saliencymap, outputmap)

        avg_psnr = psnr_sum / num_test
        avg_mae = mae_sum / num_test
        
        print('--- Evaluation on Validation Set ---')
        print('--- Average PSNR %.2f ---' % avg_psnr)
        print('--- Average MAE %.6f ---' % avg_mae)
        
		#tensorboard summaries
        validation_summary = self.sess.run(summ_eval, feed_dict={self.eva_mae: avg_mae, self.eva_psnr: avg_psnr, })
        summary_writer.add_summary(validation_summary, iter_num)
    
	#training function that will train from scratch or load the latest checkpoint and resume training
    def train(self, imgs, maps, eval_imgs, eval_maps, batch_size, epoch, lr):
        numData = len(imgs)
        numBatch = int(numData / batch_size)
		
		#loading checkpoints
        load_model_status, global_step = self.load(self.ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print('[*] Model restore success!')
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print('[*] Not find pretrained model!')
		
		#tensorboard summaries
        tf.summary.scalar('mse', self.mse)
        tf.summary.scalar('xepy', self.xepy)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter(self.log_dir + '/', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        summary_mae = tf.summary.scalar('eva_mae', self.eva_mae)
        #summary_fmeasure = tf.summary.scalar('eva_fmeasure', self.eva_fmeasure)
        merged_eva = tf.summary.merge(inputs=[summary_psnr,summary_mae])
		
        print('[*] Start training, with start epoch %d start iter %d : ' % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(eval_imgs, eval_maps, iter_num, summ_eval=merged_eva, summary_writer=writer)
        tf.get_default_graph().finalize() #after this point the tf graph will be fixed
		
		#training loop
        for epoch in xrange(start_epoch, epoch):
            print('Model: %s' % self.name)
            print(('Learning rate: {}').format(lr[epoch]))
			
			#randomizing the indices, ensures dataset shuffling at each epoch
            rand_inds = np.random.choice(numData, numData, replace=False)
			
            for batch_id in xrange(0, numBatch):
                batch_rand_inds = rand_inds[batch_id * batch_size:(batch_id + 1) * batch_size]
                img_list = [ imgs[i] for i in batch_rand_inds ] #choosing mini-batch of images
                map_list = [ maps[i] for i in batch_rand_inds ] #choosing the corresponding ground truth saliency maps
                
				#loading in float32 format, normalized to [0,1] 
                batch_images, batch_maps = load_img_map_float32_aug(img_list, map_list, self.aug_scale)
                
                feed_dict = {self.img_tensor: batch_images, self.map_tensor: batch_maps, self.lr: lr[epoch], self.is_training: True}
                self.sess.run(self.train_op, feed_dict=feed_dict) #one iteration of optimization
				
				#output loss to the terminal every 100 iterations
                if iter_num % 100 == 0:
                    feed_dict2 = {self.img_tensor: batch_images, self.map_tensor: batch_maps, self.lr: lr[epoch], self.is_training: False}
                    mse, xepy, summary = self.sess.run([self.mse, self.xepy, merged], feed_dict=feed_dict2)
                    print('Epoch: [%2d] [%4d/%4d] time: %4.4f' % (
                     epoch + 1, batch_id + 1, numBatch, time.time() - start_time))
                    print('MSE: %.6f' % mse)
                    print('X-ENTROPY: %.6f' % xepy)
                    print('\n')
                    writer.add_summary(summary, iter_num) #tensorbard summary
                if (iter_num + 1) % (numBatch // 2) == 0:
                    self.evaluate(eval_imgs, eval_maps, iter_num + 1, summ_eval=merged_eva, summary_writer=writer)
                    self.save(iter_num + 1, self.ckpt_dir)
                iter_num += 1
			
			#evaluating and saving checkpoint after each epoch
            #self.evaluate(eval_imgs, eval_maps, iter_num, summ_eval=merged_eva, summary_writer=writer)
            #self.save(iter_num, self.ckpt_dir)

        print('[*] Finish training.')

	#checkpoint saver
    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print('[*] Saving model...')
        print("\n")
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=iter_num)
		
	#checkpoint loader
    def load(self, checkpoint_dir):
        print('[*] Reading checkpoint...')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            self.saver.restore(self.sess, full_path)
            return (True, global_step)
		else:
        	return (False, 0)
	
	#testing function that will output average MAE and F-score and saves predicted saliency maps
    def test(self, test_imgs, test_maps, save_dir):
        tf.initialize_all_variables().run()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
			
		#loading checkpoints
        load_model_status, global_step = self.load(self.ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print('[*] Load weights SUCCESS...')
        
        num_test = len(test_imgs)
        mae_sum = 0
        precision_sum = 0
        recall_sum = 0
        
        print('Number of test images: {}'.format(num_test))
        for idx in xrange(num_test):
			#loading the image and map
            image_256, image_orig, size = load_images_RGB_float32(test_imgs[idx])
            map_256, map_orig = load_images_float32(test_maps[idx])
			
			#passing through network to get the saliency map prediction
            predicted_map_256= self.sess.run(self.pred_tensor_sigmoid, feed_dict={self.img_tensor: image_256, self.map_tensor: map_256, self.is_training: False})
            
			#resizing the saliency map from (256,256) back to original size
            predicted_map_256 = Image.fromarray(np.clip(255 * np.squeeze(predicted_map_256), 0, 255).astype('uint8'))
            predicted_map = predicted_map_256.resize(size, resample=Image.BICUBIC)
            predicted_map = np.expand_dims(np.expand_dims(np.array(predicted_map), axis=0), axis=3)
            
			#converting to uint8
            inputimage = np.clip(255 * image_orig, 0, 255).astype('uint8')
            saliencymap = np.clip(255 * map_orig, 0, 255).astype('uint8')
            outputmap = predicted_map
            
			#metrics
            predicted_map = predicted_map.astype(np.float32) / 255.0
            mae = cal_mae(map_orig, predicted_map)
            
            precision, recall = cal_precision_and_recall(map_orig, predicted_map)
            
            mae_sum += mae
            precision_sum += precision
            recall_sum += recall
			
			#saving the 100 images for viewing
            if idx<100:
                print('img%d, MAE: %.6f' % (idx + 1, mae))
                save_images(os.path.join(save_dir, 'test%d_concat.png' % (idx + 1)), inputimage, saliencymap, outputmap)
                
                #t = np.mean(predicted_map) * 2
                #pred_map_thr = (predicted_map>t) * 1
                #pred_map_thr = np.clip(255 * pred_map_thr, 0, 255).astype('uint8')
                #save_images(os.path.join(save_dir, 'test%d_map.png' % (idx + 1)), outputmap)
        
        
        
        avg_mae = mae_sum / num_test
        avg_precision = precision_sum / num_test
        avg_recall = recall_sum / num_test
        
        beta = 0.3
        
        if avg_precision==0 and avg_recall==0:
            f_beta=0
        elif avg_precision==1 and avg_recall==1:
            f_beta=1
        else:
            f_beta = (1+beta**2)*avg_precision*avg_recall/((beta**2)*avg_precision + avg_recall)
        print('--- Evaluation on Test Set ---')
        print('--- F-Measure %.6f ---' % f_beta)
        print('--- Average MAE %.6f ---' % avg_mae)




