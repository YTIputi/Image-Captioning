import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Dropout, GRU


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden_state):
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)  # (batch_size, 1, hidden_size)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))  # (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, 64, 1)
        context_vector = attention_weights * features  # (batch_size, 64, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, embedding_dim)
        return context_vector, attention_weights
