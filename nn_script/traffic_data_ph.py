from TensorflowToolbox.data_flow.data_ph_abs import DataPhAbs
import tensorflow as tf


class DataPh(DataPhAbs):
    def __init__(self, model_params):
        self.input_ph = tf.placeholder(tf.float32, shape=[
            model_params["batch_size"],
            model_params["feature_ph_row"],
            model_params["feature_ph_col"],
            model_params["feature_cha"]],
                                       name="feature"
                                       )

        self.label_ph = tf.placeholder(tf.float32, shape=[
            model_params["batch_size"],
            model_params["label_ph_row"],
            model_params["label_ph_col"],
            model_params["label_cha"]],
                                       name="label"
                                       )

        self.keep_prob_ph = tf.placeholder(tf.float32, 
                            name = "keep_prob")

    def get_label(self):
        return self.label_ph

    def get_input(self):
        return self.input_ph

    def get_keep_prob(self):
        return self.keep_prob_ph
    
