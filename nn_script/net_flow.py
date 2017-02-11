from traffic_data_ph import DataPh
from traffic_data_input import DataInput
from iigan_model import Model
#from vgg_atrous_model import Model
import tensorflow as tf
from TensorflowToolbox.model_flow import save_func as sf
import cv2

TF_VERSION = tf.__version__.split(".")[1]

class NetFlow(object):
    def __init__(self, model_params, load_train, load_test):
        self.load_train = load_train
        self.load_test = load_test
        self.model_params = model_params
        if load_train:
            self.train_data_input = DataInput(model_params, is_train = True)
        if load_test:
            self.test_data_input = DataInput(model_params, is_train = False)

        self.data_ph = DataPh(model_params)
        self.model = Model(self.data_ph, model_params)
        self.d_loss, self.g_loss = self.model.get_loss()
        self.d_train_op, self.g_train_op = self.model.get_train_op()

    def get_feed_dict(self, sess, is_train):
        feed_dict = dict()
        if is_train:
            input_v, label_v,file_line_v = sess.run([
                                    self.train_data_input.get_input(), 
                                    self.train_data_input.get_label(),
                                    self.train_data_input.get_file_line()])
        else:
            input_v, label_v, file_line_v = sess.run([
                                    self.test_data_input.get_input(), 
                                    self.test_data_input.get_label(),
                                    self.test_data_input.get_file_line()])

        feed_dict[self.data_ph.get_input()] = input_v
        feed_dict[self.data_ph.get_label()] = label_v

        return feed_dict

    def check_feed_dict(self, feed_dict):
        data_list = list()

        for key in feed_dict:
            data_list.append(feed_dict[key])

        cv2.imshow("image", data_list[0][0])    
        cv2.imshow("label", data_list[1][0] * 255)  
        cv2.waitKey(0)
        
    def init_var(self, sess):
        sf.add_train_var()
        sf.add_loss()
        sf.add_image("image_to_write")
        self.sum_writer = tf.summary.FileWriter(self.model_params["train_log_dir"], 
                                                sess.graph)
        self.saver = tf.train.Saver()
        self.summ = tf.summary.merge_all()

        if TF_VERSION > '11':
            init_op = tf.global_variables_initializer()
        else:
            init_op = tf.initialize_all_variables()

        sess.run(init_op)

        if self.model_params["restore_model"]:
            sf.restore_model(sess, self.saver, self.model_params["model_dir"])
        
    def mainloop(self):
        sess = tf.Session()
        self.init_var(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        if self.load_train:
            for i in range(self.model_params["max_training_iter"]):
                feed_dict = self.get_feed_dict(sess, is_train = True)
                #self.check_feed_dict(feed_dict)
                for di in range(self.model_params["d_iter"]):
                    _, d_loss_v = sess.run([self.d_train_op, 
                                self.d_loss], feed_dict)
                
                for gi in range(self.model_params["g_iter"]):
                    _, g_loss_v = sess.run([self.g_train_op,
                                self.g_loss], feed_dict)

                if i % self.model_params["test_per_iter"] == 0:
                    feed_dict = self.get_feed_dict(sess, is_train = False)
                    test_d_loss_v, test_g_loss_v = sess.run([self.d_loss, 
                                      self.g_loss], feed_dict)
                    print(("i: %d train d_loss: %.4f g_loss: %.4f " + 
                            "test d_loss: %.4f g_loss: %.4f")%(i, \
                        d_loss_v, g_loss_v, test_d_loss_v, test_g_loss_v))
                exit(1)
        else:
            pass
            #for i in range(self.model_params["test_iter"]):
            #    feed_dict = self.get_feed_dict(sess, is_train = False)
            #    loss_v = sess.run(self.loss, feed_dict)     
            #    print(loss_v)
            

        coord.request_stop()
        coord.join(threads)
