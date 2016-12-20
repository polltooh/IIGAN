import tensorflow as tf
import save_func as sf
import cv2
import tensor_data
import data_class
import model_func as mf
import os
import numpy as np
import read_proto as rp


class IIGan(object):
    def __init__(self, model_params):
    
        for key, value in model_params.iteritems():
            self.__dict__[key] = model_params[key]
        self.required_fields = {"batch_size", "iheight", "iwidth", "init_learning_rate",
                "max_training_iter", "g_iter", "d_iter", "train_log_dir", "restore_model", "model_dir",
                "file_name"}
        self.valid_field()

        #self.test_shape()
        self.map_data1_ph = tf.placeholder(tf.float32, 
                    shape = (self.batch_size, self.iheight, 
                        self.iwidth, 1),
                        name = 'map_data1_ph')

        self.image_data1_ph = tf.placeholder(tf.float32, 
                    shape = (self.batch_size, self.iheight, 
                        self.iwidth, 3),
                        name = 'image_data1_ph')

        self.map_data2_ph = tf.placeholder(tf.float32, 
                    shape = (self.batch_size, self.iheight, 
                        self.iwidth, 1),
                        name = 'map_data2_ph')

        self.image_data2_ph = tf.placeholder(tf.float32, 
                    shape = (self.batch_size, self.iheight, 
                        self.iwidth, 1), 
                        name = 'image_data2_ph')

        self.is_train_ph = tf.placeholder(tf.bool, name = 'is_train_ph')
        #self.ran_code_ph = tf.placeholder(tf.float32,
        #            shape = (self.batch_size, model_params["code_len"]), name = 'random_code')

        if not os.path.exists(self.train_log_dir):
            os.makedirs(self.train_log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model(is_train = self.is_train_ph)

        self.loss()
        self.train()
	self.data_load(self.file_name)

    def valid_field(self):
        is_valid = True
        for f in self.required_fields:
            if f not in self.__dict__:
                print(f + " is not in the model_proto")
                is_valid = False
        if not is_valid:
            exit(1)
            #assert(f in self.__dict__)

    def test_shape(self):
        input_tensor = tf.constant(1, np.float32, (self.batch_size, self.iheight, self.iwidth, 1))
        conv1 = mf.convolution_2d_layer(input_tensor, [4, 4, 1, 1], [2,2], "VALID", 0.0 , "conv1")
        conv2 = mf.convolution_2d_layer(conv1, [4, 4, 1, 1], [2,2], "VALID", 0.0 , "conv2")
        print(conv1.get_shape())
        print(conv2.get_shape())
        exit(1)

    def data_load(self, file_name):
        is_train = True

        st_data = data_class.DataClass(tf.constant([], tf.string))
        st_data.decode_class = data_class.BINClass((self.st_len,1))

        image_data = data_class.DataClass(tf.constant([], tf.string))
        image_data.decode_class = data_class.JPGClass((self.iheight, self.iwidth), 1, 0)

        tensor_list = [st_data] + [image_data]

        file_queue = tensor_data.file_queue(file_name, is_train)
        batch_tensor_list = tensor_data.file_queue_to_batch_data(file_queue, tensor_list, 
                                is_train, self.batch_size, False)

        self.st_data = batch_tensor_list[0]
        self.image_data = batch_tensor_list[1]

    def rescale_image(self, image):
        return image * 2 - 1;

    def model(self, is_train):
        wd = 0.0004
        leaky_param = 0.01
        ngf = 64 #number of filter in the first g layer
        with tf.variable_scope("G"):
            conv1 = mf.add_leaky_relu(mf.batch_norm_layer(mf.convolution_2d_layer(self.map_data1_ph,
                [4,4,1,ngf], [2,2], "VALID", wd, 'conv1'), is_train), leaky_param)
            conv1_shape = conv1.get_shape().as_list()

            conv2 = mf.add_leaky_relu(mf.batch_norm_layer(mf.convolution_2d_layer(conv1,
                [4,4,ngf,ngf*2], [2,2], "VALID", wd, 'conv2'),is_train), leaky_param)
            conv2_shape = conv2.get_shape().as_list()

            conv3 = mf.add_leaky_relu(mf.batch_norm_layer(mf.convolution_2d_layer(conv2,
                [4,4,ngf*2,ngf*4], [2,2], "VALID", wd, 'conv3'), is_train), leaky_param)
            conv3_shape = conv3.get_shape().as_list()

            conv4 = mf.add_leaky_relu(mf.batch_norm_layer(mf.convolution_2d_layer(conv3,
                [4,4,ngf*4,ngf*8], [2,2], "VALID", wd, 'conv4'), is_train), leaky_param)
            conv4_shape = conv4.get_shape().as_list()

            deconv3_shape = conv3_shape
            deconv3_shape[3] = ngf*4
            deconv3 = mf.add_leaky_relu(mf.batch_norm_layer(mf.deconvolution_2d_layer(conv4, 
                [4,4,ngf*4,ngf*8], [2,2], deconv3_shape, "VALID", wd, "deconv3"), is_train), leaky_param)
            deconv3 = tf.concat(3, [deconv3, conv3])

            deconv2_shape = conv2_shape
            deconv2_shape[3] = ngf*4
            deconv2 = mf.add_leaky_relu(mf.batch_norm_layer(mf.deconvolution_2d_layer(deconv3, 
                [4,4,ngf*4,ngf*8], [2,2], deconv2_shape, "VALID", wd, "deconv2"), is_train), leaky_param)
            deconv2 = tf.concat(3, [deconv2, conv2])

            deconv1_shape = conv1_shape
            deconv1_shape[3] = ngf*4
            deconv1 = mf.add_leaky_relu(mf.batch_norm_layer(mf.deconvolution_2d_layer(deconv2, 
                [4,4,ngf*4,384], [2,2], deconv1_shape, "VALID", wd, "deconv1"), is_train), leaky_param)
            deconv1 = tf.concat(3, [deconv1, conv1])

            deconv0_shape = self.image_data1_ph.get_shape().as_list()
            deconv0_shape[3] = 3
            deconv0 = mf.add_leaky_relu(mf.batch_norm_layer(mf.deconvolution_2d_layer(deconv1, 
                [4,4,3,320], [2,2], deconv0_shape, "VALID", wd, "deconv0"), is_train), leaky_param)

            self.g_image = tf.tanh(deconv0, "g_image")

        image_write = tf.concat(2, [tf.tile(self.map_data1_ph,[1,1,1,3]), self.g_image, self.image_data1_ph])
        tf.add_to_collection("image_to_write", image_write)

        with tf.variable_scope("D"):
            fake_concat = tf.concat(3, [self.g_image, self.image_data1_ph])
            real_concat = tf.concat(3, [self.map_data2_ph, self.image_data2_ph])
            fake_real_concat = tf.concat(0, [fake_concat, real_concat])
            #conv1 = mf.convolution_2d_layer(self.image_data_ph, [5, 5, 2, 64], [2,2], "VALID", wd, "conv1")
            conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(fake_real_concat, [3, 3, 4 * 2, 32], \
                    [2,2], "VALID", wd, "conv1"), leaky_param)
            #conv1_maxpool = mf.maxpool_2d_layer(conv1, [2,2], [2,2], "maxpool1")

            conv2 = mf.add_leaky_relu(mf.convolution_2d_layer(conv1, [3, 3, 32, 64], [2,2], "VALID", wd, "conv2"), leaky_param)
            #conv2_maxpool = mf.maxpool_2d_layer(conv2, [2,2], [2,2], "maxpool2")

            conv3 = mf.add_leaky_relu(mf.convolution_2d_layer(conv2, [3, 3, 64, 128], [2,2], "VALID", wd, "conv3"), leaky_param)
            #conv3_maxpool = mf.maxpool_2d_layer(conv3, [2,2], [2,2], "maxpool3")

            conv4 = mf.add_leaky_relu(mf.convolution_2d_layer(conv3, [3, 3, 128, 128], [2,2], "VALID", wd, "conv4"), leaky_param)

            conv5 = mf.add_leaky_relu(mf.convolution_2d_layer(conv3, [3, 3, 128, 256], [2,2], "VALID", wd, "conv5"), leaky_param)
            #conv4_maxpool = mf.maxpool_2d_layer(conv4, [2,2], [2,2], "maxpool4")
            self.fc = mf.fully_connected_layer(conv5, 1, wd, "fc")

        #decode network
        #with tf.variable_scope("C"):
        #    conv1 = mf.add_leaky_relu(mf.convolution_2d_layer(self.g_image, [5, 5, 1, 64], [2,2], "VALID", wd, "conv1"), leaky_param)
        #    conv2 = mf.add_leaky_relu(mf.convolution_2d_layer(conv1, [5, 5, 64, 128], [2,2], "VALID", wd, "conv2"), leaky_param)
        #    conv3 = mf.add_leaky_relu(mf.convolution_2d_layer(conv2, [5, 5, 128, 512], [2,2], "VALID", wd, "conv3"), leaky_param)
        #    conv4 = mf.add_leaky_relu(mf.convolution_2d_layer(conv3, [5, 5, 512, 128], [2,2], "VALID", wd, "conv4"), leaky_param)
        #    self.st_infer = mf.fully_connected_layer(conv4, self.st_len, wd, "fc")

    def loss(self):
        real_label = tf.constant(1, dtype = tf.float32, shape = (self.batch_size, 1))
        fake_label = tf.constant(0, dtype = tf.float32, shape = (self.batch_size, 1))

        fake_real_label = tf.concat(0, [fake_label, real_label])
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fc, fake_real_label), name = "d_loss")
        tf.add_to_collection("losses", self.d_loss)
        # switch label order
        real_fake_label = tf.concat(0, [real_label, fake_label])
        g_total_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.fc, real_fake_label)
        g_lambda = 1.0

        self.g_loss = tf.reduce_mean(g_lambda * g_total_loss[:self.batch_size,:], name = "g_loss")
        tf.add_to_collection("losses", self.g_loss)
        
        #self.c_loss = mf.l2_loss(self.st_infer, tf.squeeze(self.st_data_ph), "MEAN", "c_loss")
        #tf.add_to_collection("losses", self.c_loss)


    def train(self):
        d_vars = [v for v in tf.trainable_variables() if "D" in v.op.name]
        g_vars = [v for v in tf.trainable_variables() if "G" in v.op.name]
        #c_vars = [v for v in tf.trainable_variables() if "C" in v.op.name]

        self.d_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.d_loss, var_list = d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.g_loss, var_list = g_vars)
        #self.c_optim = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.c_loss, var_list = g_vars + c_vars)


    def mainloop(self):
        sess = tf.Session()
        
        sf.add_train_var()
        sf.add_loss()
        sf.add_image("image_to_write")
        sum_writer = tf.summary.FileWriter(self.train_log_dir, sess.graph)

        saver = tf.train.Saver()
        summ = tf.merge_all_summaries()

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        if self.restore_model:
            sf.restore_model(sess, saver, self.model_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord, sess = sess)

        for i in xrange(self.max_training_iter):
            train_st_data_v, train_image_data_v = sess.run([self.st_data, self.image_data])
            #train_image_data_v = np.random.uniform(-1, 1, (self.batch_size, self.iheight, self.iwidth, 1))
            #train_st_data_v = np.random.uniform(-1, 1, (self.batch_size, self.st_len))

            ran_code = np.random.uniform(-1, 1, size = (self.batch_size, 100))
            train_image_data_v = self.rescale_image(train_image_data_v)

            feed_data = {self.image_data_ph: train_image_data_v,
                                            self.st_data_ph: train_st_data_v,
                                            self.ran_code_ph:ran_code}

            for di in xrange(self.d_iter):
                _, d_loss_v = sess.run([self.d_optim, self.d_loss], feed_dict = feed_data)
            for gi in xrange(self.g_iter):
                #_, _, g_image_v, g_loss_v, c_loss_v, summ_v = sess.run([self.g_optim, self.c_optim, self.g_image, 
                #                            self.g_loss, self.c_loss, summ], feed_dict = feed_data)
                _, g_image_v, g_loss_v, summ_v = sess.run([self.g_optim, self.g_image, 
                                            self.g_loss, summ], feed_dict = feed_data)

            if i%100 == 0:
                sum_writer.add_summary(summ_v, i)
                print("iter: %d, d_loss: %.3f, g_loss: %.3f"%(i, d_loss_v, g_loss_v))

            if i != 0 and (i %1000 == 0 or i == self.max_training_iter - 1):
                sf.save_model(sess, saver, self.model_dir, i)
