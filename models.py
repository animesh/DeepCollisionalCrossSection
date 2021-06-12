import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np



def BiRNN_new(x, c, l, num_layers, num_hidden, meta_data,
              num_classes, timesteps, keep_prob, uncertainty, 
              is_train=True, use_sequence_lengths=False, use_embedding=True,
              sequence_length=66, dict_size=32):

        if use_embedding:
            emb = tf.keras.layers.Embedding(dict_size, num_hidden, input_length=sequence_length)
            x = emb(x)

        num_layers=num_layers
        with tf.name_scope("birnn"):
            #lstm_fw_cell = [tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(num_hidden), input_keep_prob = keep_prob) for _ in range(num_layers)]
            #lstm_bw_cell = [tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(num_hidden), input_keep_prob = keep_prob) for _ in range(num_layers)]
            #lstm_fw_cell = [rnn.ConvLSTMCell(1,[66,32],128, (5,1)) for _ in range(num_layers)]	
            #lstm_bw_cell = [rnn.ConvLSTMCell(1,[66,32],128, (5,1)) for _ in range(num_layers)]
            lstm_fw_cell = [rnn.BasicLSTMCell(num_hidden) for _ in range(num_layers)]
            lstm_bw_cell = [rnn.BasicLSTMCell(num_hidden) for _ in range(num_layers)]
            #

        if use_sequence_lengths:
            rnn_outputs_all, final_fw, final_bw = rnn.stack_bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, x, sequence_length=l, dtype=tf.float32)
        else:
            rnn_outputs_all, final_fw, final_bw = rnn.stack_bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
            final_fw = final_fw[-1].h
            final_bw = final_bw[-1].h
		
		
        out = tf.concat([final_fw, final_bw], 1)
        print(out, final_bw, final_fw)
        feat = tf.concat([out, c], axis=1)

        #add another layer !
        l1 = tf.contrib.slim.fully_connected(feat, num_hidden)
        # l1 = tf.layers.dropout(l1, rate=keep_prob, training=is_train)

        # l2 = tf.contrib.slim.fully_connected(l1, num_hidden)
        preds = tf.contrib.slim.fully_connected(l1, num_classes, activation_fn=None)
        # maybe we need to return somethin for attention score
        return preds, l1, None, None, None, None

