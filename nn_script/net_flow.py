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
        self.d_clip = self.model.get_clip()

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
        #normalize to [-1, 1]

        input_v = (input_v - 0.5) * 2
        label_v = (label_v * 100 - 0.5) * 2
        feed_dict[self.data_ph.get_input()] = input_v
        feed_dict[self.data_ph.get_label()] = label_v
        feed_dict[self.data_ph.get_keep_prob()] = 0.5

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
        self.saver = tf.train.Saver()

        if TF_VERSION > '11':
            self.sum_writer = tf.summary.FileWriter(self.model_params["train_log_dir"], 
                                         sess.graph)
            self.summ = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
        else:
            self.sum_writer = tf.train.SummaryWritter(self.model_params["train_log_dir"], 
                                         sess.graph)
            self.summ = tf.merge_all_summaries()
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
                    _, d_loss_v, _ = sess.run([self.d_train_op, 
                                self.d_loss, self.d_clip], feed_dict)
                
                for gi in range(self.model_params["g_iter"]):
                    _, g_loss_v = sess.run([self.g_train_op,
                                self.g_loss], feed_dict)

                if i % self.model_params["test_per_iter"] == 0:
                    feed_dict = self.get_feed_dict(sess, is_train = False)
                    test_d_loss_v, test_g_loss_v , summ_v = sess.run([self.d_loss, 
                                      self.g_loss, self.summ], feed_dict)
                    print(("i: %d train d_loss: %.8f g_loss: %.8f " + 
                            "test d_loss: %.8f g_loss: %.8f")%(i, \
                        d_loss_v, g_loss_v, test_d_loss_v, test_g_loss_v))

                self.sum_writer.add_summary(summ_v, i)

                if i != 0 and (i % self.model_params["save_per_iter"] == 0 or \
                                i == self.model_params["max_training_iter"] - 1):
                    sf.save_model(sess, self.saver, self.model_params["model_dir"], i)

        else:
            pass
            #for i in range(self.model_params["test_iter"]):
            #    feed_dict = self.get_feed_dict(sess, is_train = False)
            #    loss_v = sess.run(self.loss, feed_dict)     
            #    print(loss_v)
            

        coord.request_stop()
        coord.join(threads)
