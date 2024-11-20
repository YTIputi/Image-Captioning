import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Dropout, GRU
import BahdanauAttention

class DecoderWithAttention(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(DecoderWithAttention, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden_state):
        # Attention
        context_vector, attention_weights = self.attention(features, hidden_state)

        # Embedding
        x = self.embedding(x)  # (batch_size, 1, embedding_dim)

        # Concatenate context_vector + embedding
        x = tf.concat([tf.expand_dims(context_vector, 1), x],
                      axis=-1)  # (batch_size, 1, embedding_dim + context_vector_dim)

        # GRU
        output, state = self.gru(x)

        # Output layer
        x = self.fc(output)  # (batch_size, 1, vocab_size)
        x = tf.squeeze(x, axis=1)  # Remove time dimension

        return x, state, attention_weights
