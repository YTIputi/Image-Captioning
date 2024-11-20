from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow as tf

# Токенизация текста с токенами <start> и <end>
captions = ['<start> this is a sample caption <end>', '<start> another caption <end>']
tokenizer = Tokenizer(num_words=5000, oov_token='<unk>')
tokenizer.fit_on_texts(captions)

# Преобразование текста в последовательности
sequences = tokenizer.texts_to_sequences(['<start> this is a sample caption <end>'])
print("Токенизированные последовательности:", sequences)

# Декодер с LSTM
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)

    def call(self, x, hidden_state):
        # Преобразуем токены в векторы
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=[hidden_state, hidden_state])
        x = self.fc(output)
        return x, state_h, state_c

# Пример использования декодера
vocab_size = len(tokenizer.word_index) + 1  # Учитываем количество уникальных токенов
embedding_dim = 256
units = 512
decoder = Decoder(vocab_size, embedding_dim, units)

# Входные данные
if '<start>' in tokenizer.word_index:
    input_sequence = tf.expand_dims([tokenizer.word_index['<start>']], 0)  # Начальный токен
    hidden_state = tf.zeros((1, units))

    output, state_h, state_c = decoder(input_sequence, hidden_state)
    print("Размер выходных данных декодера:", output.shape)  # Размер выходных данных
else:
    print("Ошибка: токен '<start>' отсутствует в словаре токенизатора.")
