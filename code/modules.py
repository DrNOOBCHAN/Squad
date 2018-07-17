# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


def batch_matmul(matrix, weight):
	"""
	Note: tf.matmul is only for 2D in lower tensorflow version
	Have to define our matmul for 3D batch_size included matmul

	Inputs:
	 matrix: The one with batch_size (3D)
	 weight: 2D matrix to be multiplied
	"""

	#Examples are from (question_hiddens, ) as Inputs
	matrix_shape = matrix.get_shape().as_list()
	weight_shape = weight.get_shape().as_list()
	matrix_2D = tf.reshape(matrix, [-1, matrix_shape[-1]]) #e.g. (batch_size * hidden_size, hidden_size)
	product = tf.reshape(tf.matmul(matrix_2D, weight), [-1, matrix_shape[1], weight_shape[-1]]) #e.g. (batch_size, question_len, hidden_size * 2)

	return product


class Output_Layer(object):
    def __init__(self, question_hidden_size, question_len, context_len, prev_hidden_size, output_hidden_size):
        """
        Inputs:
            prev_hidden_size: size of the hidden state in Self Attention layer output h_tP
        """
        self.question_hidden_size = question_hidden_size
        self.question_len = question_len
        self.context_len = context_len
        self.prev_hidden_size = prev_hidden_size
        self.output_hidden_size = output_hidden_size
        
    def build_graph(self, question_hiddens, inputs, context_mask, qn_mask):
        """
        Inputs:
            question_hiddens: uQ (batch_size, question_len, question_hidden_size)
            inputs: self attention layer output h_P (batch_size, context_len, prev_hidden_size)
            context_mask: mask for context that 1s for real words, 0s for padding (batch_size, context_len)
            qn_mask: mask for question that 1s for real words, 0s for padding (batch_size, question_len)
        """
        with vs.variable_scope('output'):
            #rQ part
            W_uQ = tf.get_variable('W_uQ', shape=[self.question_hidden_size, self.output_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=36))
            W_vQ = tf.get_variable('W_vQ', shape=[self.prev_hidden_size, self.output_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=36))
            V_rQ = tf.get_variable('V_rQ', shape=[self.question_len, self.prev_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=63))
            V = tf.get_variable('V', shape=[self.output_hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer(seed=63))
            
            ###Initialise h_0 = rQ for the rnn initial_state
            rQ_result1 = batch_matmul(question_hiddens, W_uQ) #(batch_size, question_len, output_hidden_size)
            rQ_result2 = tf.expand_dims(tf.matmul(V_rQ, W_vQ), 0) #(1, question_len, output_hidden_size)
            z_rQ = tf.nn.tanh(rQ_result1 + rQ_result2) #(batch_size, question_len, output_hidden_size)
            s_rQ = batch_matmul(z_rQ, V) #(batch_size, question_len, 1)
            
            s_rQ = tf.reshape(s_rQ, [-1, self.question_len]) #(batch_size, question_len)
            
            
            _ , a_rQ = masked_softmax(s_rQ, qn_mask, 1) #(batch_size, question_len)
            
            a_rQ = tf.expand_dims(a_rQ, 1)
            rQ = tf.matmul(a_rQ, question_hiddens) #(batch_size, 1, question_hidden_size)
            
            rQ = tf.reshape(rQ, [-1, self.question_hidden_size]) #(batch_size, question_hidden_size)
            
            ###pointer network first part
            W_hP = tf.get_variable('W_hP', shape=[self.prev_hidden_size, self.output_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=60))
            W_ha = tf.get_variable('W_ha', shape=[self.question_hidden_size, self.output_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=19))
            V1 = tf.get_variable('V1', shape=[self.output_hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer(seed=92))
            
            
            s_t1_fir = batch_matmul(inputs , W_hP) #(batch_size, context_len, output_hidden_size)
            s_t1_sec = tf.matmul(rQ, W_ha) #(batch_size, output_hidden_size)
            s_t1_sec = tf.expand_dims(s_t1_sec, 1) #(batch_size, 1, output_hidden_size)
            
            z1 = tf.nn.tanh(s_t1_fir + s_t1_sec) #(batch_size, context_len, output_hidden_size)
            s_t1 = batch_matmul(z1, V1) #(batch_size, context_len, 1)
            s_t1 = tf.reshape(s_t1, [-1, self.context_len])
            
            p1 , a_t1 = masked_softmax(s_t1, context_mask, 1) #(batch_size, context_len)
            
            #first prediction
            #p1 = tf.argmax(a_t1, 1) #(batch_size) array
            
            #prepare input to rnn
            a_t1 = tf.expand_dims(a_t1, 1) #(batch_size, 1, context_len)
            c_1 = tf.matmul(a_t1, inputs) #(batch_size, 1, prev_hidden_size)
            c_1 = tf.reshape(c_1, [-1, self.prev_hidden_size])
            
            #pass to RNN
            with vs.variable_scope('output_t1'):
            	cell = tf.contrib.rnn.BasicRNNCell(num_units=self.question_hidden_size)
            	_, h1 = cell(c_1, rQ) #(batch_size, question_hidden_size)
            
            ###pointer network second part
            s_t2_fir = s_t1_fir #(batch_size, context_len, output_hidden_size)
            s_t2_sec = tf.matmul(h1, W_ha) #(batch_size, output_hidden_size)
            s_t2_sec = tf.expand_dims(s_t2_sec, 1) #(batch_size, 1, output_hidden_size)
            
            z2 = tf.nn.tanh(s_t2_fir + s_t2_sec) #(batch_size, context_len, output_hidden_size)
            s_t2 = batch_matmul(z2, V1) #(batch_size, context_len, 1)
            s_t2 = tf.reshape(s_t2, [-1, self.context_len])
            
            p2, a_t2 = masked_softmax(s_t2, context_mask, 1) #(batch_size, context_len)
            #first prediction
            #p2 = tf.argmax(a_t2, 1) #(batch_size) array
            
            """
            #prepare input to rnn
            a_t2 = tf.expand_dims(a_t2, 1) #(batch_size, 1, context_len)
            c_2 = tf.matmul(a_t2, inputs) #(batch_size, 1, prev_hidden_size)
            c_2 = tf.reshape(c_2, [-1, self.prev_hidden_size])
            

            #pass to RNN
            with vs.variable_scope('output_t2'):
            	cell = tf.contrib.rnn.BasicRNNCell(num_units=self.question_hidden_size)
            	_, h2 = cell(c_2, rQ) #(batch_size, question_hidden_size)
            """

            return p1, p2


class SelfAttn(object):
    def __init__(self, content_len, prev_hidden_size, output_hidden_size):
        """
        Inputs:
            content_len: Number of words in context
            prev_hidden_size: The size of (v_tP) from Gated Attention RNN layer
            output_hidden_size: The size of output of Self Attention layer (h_t)
        """
        self.content_len = content_len
        self.prev_hidden_size = prev_hidden_size
        self.output_hidden_size = output_hidden_size
    
    def build_graph(self, inputs, values_mask):
        """
        Inputs:
         inputs: states from Gated Attention RNN layer (batch_size, content_len, prev_hidden_size)
         values_mask: mask that stores 1 for real words, 0 for padding (batch_size, content_len)
        """
        with vs.variable_scope('self_attention'):
            W_1 = tf.get_variable('W_1', shape=[self.prev_hidden_size, self.output_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=33))
            W_2 = tf.get_variable('W_2', shape=[self.prev_hidden_size, self.output_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=80))
            V = tf.get_variable('V', shape=[self.output_hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer(seed=54))

            W_1_result = tf.expand_dims(batch_matmul(inputs, W_1), 1)
            W_2_result = tf.expand_dims(batch_matmul(inputs, W_2), 2)

            z = tf.nn.tanh(W_1_result + W_2_result) #(batch_size, content_len, content_len, output_hidden_size)
            s_t = tf.reshape(z, [-1, self.content_len * self.content_len, self.output_hidden_size]) #(batch_size, content_len * content_len, output_hidden_size)
            s_t = batch_matmul(s_t, V) #(batch_size, content_len*content_len, 1)
            s_t = tf.reshape(s_t, [-1, self.content_len, self.content_len]) #(batch_size, content_len, content_len)

            self_attn_mask = tf.expand_dims(values_mask, 1)
            _, a_t = masked_softmax(s_t, self_attn_mask, 2) #(batch_size, content_len, content_len)

            c_t = tf.matmul(a_t, inputs) 
            
            return a_t, c_t


class GateAttnCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, content_hidden_size, question_hidden_size, hidden_size, question_hiddens):
        self.content_hidden_size = content_hidden_size
        self.question_hidden_size = question_hidden_size
        self.hidden_size = hidden_size
        self.uQ = question_hiddens
        self.cell = tf.contrib.rnn.GRUCell(hidden_size)
        
    @property
    def state_size(self):
        return self.hidden_size
    
    @property
    def output_size(self):
        return self.hidden_size
    
    def __call__(self, inputs, state):
        W_uQ = tf.get_variable('W_uQ', shape=[self.question_hidden_size, self.hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=8))
        W_uP = tf.get_variable('W_uP', shape=[self.content_hidden_size, self.hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=79))
        W_vP = tf.get_variable('W_vP', shape=[self.hidden_size, self.hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=57))
        W_g = tf.get_variable('W_g', shape=[self.content_hidden_size + self.question_hidden_size, self.content_hidden_size + self.question_hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=45))
        V = tf.get_variable('V', shape=[self.hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer(seed=45))
        
        W_uQ_uQ = batch_matmul(self.uQ, W_uQ)
        W_uP_utP = tf.expand_dims(tf.matmul(inputs, W_uP), 1) #(1, 1, hidden_size)

        W_vP_vtP = tf.expand_dims(tf.matmul(state, W_vP), 1) #(1, 1, hidden_size)
        z = tf.nn.tanh(W_uQ_uQ + W_uP_utP + W_vP_vtP) #e.g. (batch_size, question_len, hidden_size)
        s_t = batch_matmul(z, V) # (batch_size, question_len, 1)
        a_t = tf.nn.softmax(s_t, 1) # (batch_size, question_len, 1)
        c_t = tf.reduce_mean(tf.multiply(a_t, self.uQ), 1) #(batch_size, hidden_size)
        concat = tf.concat([inputs, c_t], 1) #(batch_size, hidden_size + content_hidden_size)
        g_t = tf.nn.sigmoid(tf.matmul(concat,W_g)) * concat
        
        output, new_state = self.cell(g_t, state)
        return output, new_state

class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
