from TensorflowToolbox.model_flow.model_abs import ModelAbs
from TensorflowToolbox.model_flow import model_func as mf
import tensorflow as tf

class Model(ModelAbs):
    def __init__(self, data_ph, model_params):
        self.model_infer(data_ph, model_params)
        self.model_loss(data_ph, model_params)
        self.model_mini(model_params)
   
    def conv2_wapper(self, input_tensor, output_dim, wd, layer_name, \
                    add_relu, leaky_param):
        kernel_shape = [4, 4, input_tensor.get_shape().as_list()[3], output_dim]

        conv = mf.convolution_2d_layer(input_tensor, kernel_shape, [2,2], 
                    "VALID", wd, layer_name)
        if add_relu:
            conv = mf.add_leaky_relu(conv, leaky_param)

        return conv

    def deconv2_wapper(self, input_tensor, output, wd, layer_name, 
                    add_relu, leaky_param, concat):
        """output could be the tensor or the shape list"""
        if type(output) is not list:
            output_shape = output.get_shape().as_list()
        else:
            output_shape = output
      
        deconv = mf.deconvolution_2d_layer(input_tensor, 
                [4, 4, output_shape[3], input_tensor.get_shape().as_list()[3]],
                [2,2], output_shape, "VALID", 
                wd, layer_name)

        if add_relu:
            deconv = mf.add_leaky_relu(deconv, leaky_param)

        if concat:
            deconv = tf.concat(3, [deconv, output], 
                        deconv.op.name + "_concat")

        return deconv

    def keep_prob_wapper(self, input_tensor, keep_prob):
        input_tensor = tf.nn.dropout(input_tensor, 
                        keep_prob, name = input_tensor.op.name + "_dropout")
        return input_tensor
        
        
    def generator(self, data_ph, wd, leaky_param):
        image = data_ph.get_input()
        keep_prob = data_ph.get_keep_prob()
        output_c_dim = data_ph.get_label().get_shape().as_list()[3]

        gf_dim = 64 # the number of channel of the first layer
        with tf.variable_scope("G"):
            # encode network
            e1 = self.conv2_wapper(image, gf_dim, wd, "e1", True, leaky_param)
            print(e1)
            e2 = self.conv2_wapper(e1, gf_dim * 2, wd, "e2", True, leaky_param)
            print(e2)
            e3 = self.conv2_wapper(e2, gf_dim * 4, wd, "e3", True, leaky_param)
            print(e3)
            e4 = self.conv2_wapper(e3, gf_dim * 8, wd, "e4", True, leaky_param)
            print(e4)
            e5 = self.conv2_wapper(e4, gf_dim * 8, wd, "e5", True, leaky_param)
            print(e5)
            e6 = self.conv2_wapper(e5, gf_dim * 8, wd, "e6", True, leaky_param)
            print(e6)
            
            # decode network
            d1 = self.deconv2_wapper(e6, e5, wd, "d1", True, leaky_param, True)
            d1 = self.keep_prob_wapper(d1, keep_prob)
            print(d1)
            d2 = self.deconv2_wapper(d1, e4, wd, "d2", True, leaky_param, True)
            d2 = self.keep_prob_wapper(d2, keep_prob)
            print(d2)
            d3 = self.deconv2_wapper(d2, e3, wd, "d3", True, leaky_param, True)
            d3 = self.keep_prob_wapper(d3, keep_prob)
            print(d3)
            d4 = self.deconv2_wapper(d3, e2, wd, "d4", True, leaky_param, True)
            print(d4)
            d5 = self.deconv2_wapper(d4, e1, wd, "d5", True, leaky_param, True)
            print(d5)

            output_shape = image.get_shape().as_list()
            output_shape[3] = output_c_dim
            d6 = self.deconv2_wapper(d5, output_shape, wd, "d6", 
                            False, leaky_param, False)

            g_image = tf.tanh(d6, d6.op.name + "_tanh")
            print(g_image)

        return g_image 

    def discriminator(self, g_image, data_ph, wd, leaky_param):
        df_dim = 64 # first dimention of the first layer
        with tf.variable_scope("D"):
            fake_concat = tf.concat(3, [self.g_image, data_ph.get_input()])
            real_concat = tf.concat(3, [data_ph.get_label(), data_ph.get_input()])
            fake_real_concat = tf.concat(0, [fake_concat, real_concat])
            c1 = self.conv2_wapper(fake_real_concat, df_dim, wd, "c1", 
                                    True, leaky_param)
            print(c1)
            c2 = self.conv2_wapper(c1, df_dim * 2, wd, "c2", True, leaky_param)
            print(c2)
            c3 = self.conv2_wapper(c2, df_dim * 4, wd, "c3", True, leaky_param)
            print(c3)
            c4 = self.conv2_wapper(c3, df_dim * 8, wd, "c4", True, leaky_param)
            print(c4)
            c5 = self.conv2_wapper(c4, df_dim * 8, wd, "c5", True, leaky_param)
            print(c5)
            c6 = self.conv2_wapper(c5, df_dim * 8, wd, "c6", True, leaky_param)
            print(c6)
            fc = mf.fully_connected_layer(c6, 1, wd, "fc")
            print(fc)
        return fc

    def model_infer(self, data_ph, model_params):

        leaky_param = model_params["leaky_param"]
        wd = model_params["weight_decay"]

        g_image = self.generator(data_ph, wd, leaky_param)
        tf.add_to_collection("image_to_write", g_image)
        tf.add_to_collection("image_to_write", data_ph.get_input())
        tf.add_to_collection("image_to_write", data_ph.get_label())
        
        self.g_image = g_image
        fc = self.discriminator(g_image, data_ph, wd, leaky_param)
        self.fc = fc

    def model_loss_org(self, data_ph, model_params):
        wd_loss = tf.add_n(tf.get_collection("losses"), name = 'wd_loss')
        batch_size = data_ph.get_input().get_shape().as_list()[0]

        real_label = tf.constant(1, dtype = tf.float32, shape = (batch_size, 1))
        fake_label = tf.constant(0, dtype = tf.float32, shape = (batch_size, 1))
        fake_real_label = tf.concat(0, [fake_label, real_label])

        self.d_loss = tf.reduce_mean(
                     tf.nn.sigmoid_cross_entropy_with_logits(self.fc, 
                     fake_real_label), name = "d_xentropy_loss")
        tf.add_to_collection("losses", self.d_loss)

        self.d_w_loss = tf.add(self.d_loss, wd_loss,
                     name = 'd_loss')
        tf.add_to_collection("losses", self.d_w_loss)

        real_fake_label = tf.concat(0, [real_label, fake_label])
        g_total_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.fc, 
                                                    real_fake_label)

        self.g_loss = tf.reduce_mean(g_total_loss[:batch_size,:], 
                        name = "g_xentropy_loss")
        tf.add_to_collection("losses", self.g_loss)

        self.g_w_loss = tf.add(self.g_loss, wd_loss,
                        name = "g_loss")
        tf.add_to_collection("losses", self.g_w_loss)
    
    def loss_wapper(self, infer, label):
        # cross entropy without log and without sigmoid function
        return (1 - 2 * label) * infer

    def model_loss(self, data_ph, model_params):
        # refer to Wasserstein GAN training percedure
        wd_loss = tf.add_n(tf.get_collection("losses"), name = 'wd_loss')
        batch_size = data_ph.get_input().get_shape().as_list()[0]

        real_label = tf.constant(1, dtype = tf.float32, shape = (batch_size, 1))
        fake_label = tf.constant(0, dtype = tf.float32, shape = (batch_size, 1))
        fake_real_label = tf.concat(0, [fake_label, real_label])

        self.d_loss = tf.reduce_mean(self.loss_wapper(self.fc, fake_real_label),
                    name = "d_fc_loss")

        tf.add_to_collection("losses", self.d_loss)

        self.d_w_loss = tf.add(self.d_loss, wd_loss,
                     name = 'd_loss')
        tf.add_to_collection("losses", self.d_w_loss)


        real_fake_label = tf.concat(0, [real_label, fake_label])

        g_total_loss = self.loss_wapper(self.fc, real_fake_label)

        self.g_loss = tf.reduce_mean(g_total_loss[:batch_size,:], 
                        name = "g_fc_loss")

        tf.add_to_collection("losses", self.g_loss)

        self.g_w_loss = tf.add(self.g_loss, wd_loss,
                        name = "g_loss")

        tf.add_to_collection("losses", self.g_w_loss)

    def model_mini(self, model_params):
        d_vars = [v for v in tf.trainable_variables() if "D" in v.op.name]
        g_vars = [v for v in tf.trainable_variables() if "G" in v.op.name]

        self.d_optim = tf.train.RMSPropOptimizer(
                            model_params["init_learning_rate"]).minimize(
                            self.d_w_loss, var_list = d_vars)
        #self.d_optim = tf.train.AdamOptimizer(
        #                    model_params["init_learning_rate"]).minimize(
        #                    self.d_w_loss, var_list = d_vars)

        self.g_optim = tf.train.RMSPropOptimizer(
                            model_params["init_learning_rate"]).minimize(
                            self.g_w_loss, var_list = g_vars)
    
        self.clip_d = [d.assign(tf.clip_by_value(d, -0.01, 0.01)) for d in d_vars]

    def get_train_op(self):
        return self.d_optim, self.g_optim
        
    def get_loss(self):
        return self.d_loss, self.g_loss

    def get_clip(self):
        return self.clip_d

        

