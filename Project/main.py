from tensorflow.keras.preprocessing.text import Tokenizer
import DecoderWithAttention
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from DecoderWithAttention import DecoderWithAttention
import pandas as pd
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Main:
    def __init__(self, embedding_dim, units, vocab_size, max_length):
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.decoder = DecoderWithAttention(embedding_dim, units, vocab_size)
        self.encoder = self.build_encoder()

    def main(self, csv_path, image_dir):
        df = pd.read_csv(csv_path, delimiter="|")
        captions = df[' comment'].values
        image_paths = [os.path.join(image_dir, img) for img in df['image_name'].values]

        # Токенизация
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<unk>')
        tokenizer.fit_on_texts(captions)
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        # Создание последовательностей
        sequences = tokenizer.texts_to_sequences(captions)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        start_token, end_token = tokenizer.word_index['<start>'], tokenizer.word_index['<end>']

        # Обучение модели
        self.train(image_paths, padded_sequences, tokenizer, start_token, end_token)


    # Пример: загрузка изображения
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img


    def generate_caption(self, image, encoder, decoder, tokenizer, max_length):
        features = encoder.predict(image)  # (1, embedding_dim)
        hidden_state = tf.zeros((1, decoder.units))

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for _ in range(max_length):
            predictions, hidden_state, _ = decoder(dec_input, features, hidden_state)
            predicted_id = tf.argmax(predictions[0]).numpy()

            word = tokenizer.index_word.get(predicted_id, '<unk>')
            result.append(word)

            if word == '<end>':
                break

            dec_input = tf.expand_dims([predicted_id], 0)

        return ' '.join(result)


    def build_encoder(self):
        base_model = InceptionV3(weights='imagenet')
        base_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

        for layer in base_model.layers:
            layer.trainable = False

        return base_model
    
    def train(self, image_paths, captions, tokenizer, start_token, end_token, batch_size=64, epochs=20):
        dataset = self.create_dataset(image_paths, captions, batch_size, tokenizer)
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(epochs):
            total_loss = 0
            for batch, (img_tensor, target) in enumerate(dataset):
                loss = self.train_step(img_tensor, target, tokenizer, start_token, end_token, optimizer)
                total_loss += loss

                if batch % 100 == 0:
                    print(f'Epoch {epoch + 1}, Batch {batch}, Loss {loss.numpy()}')

            print(f'Epoch {epoch + 1}, Loss {total_loss / batch}')

    def create_dataset(self, image_paths, captions, batch_size, tokenizer):
        def load_data(image_path, caption):
            img_tensor = self.encoder.predict(self.preprocess_image(image_path))
            return img_tensor, caption

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
        dataset = dataset.map(lambda img, cap: tf.py_function(load_data, [img, cap], [tf.float32, tf.int32]))
        dataset = dataset.shuffle(len(image_paths)).batch(batch_size, drop_remainder=True)
        return dataset

    def train_step(self, img_tensor, target, tokenizer, start_token, end_token, optimizer):
        loss = 0
        hidden_state = tf.zeros((img_tensor.shape[0], self.units))
        dec_input = tf.expand_dims([start_token] * img_tensor.shape[0], 1)

        with tf.GradientTape() as tape:
            for t in range(1, target.shape[1]):
                predictions, hidden_state, _ = self.decoder(dec_input, img_tensor, hidden_state)
                loss += self.loss_function(target[:, t], predictions)

                dec_input = tf.expand_dims(target[:, t], 1)

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss / int(target.shape[1])

    @staticmethod
    def loss_function(real, pred):
        mask = tf.math.not_equal(real, 0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)
        loss = tf.reduce_mean(loss * tf.cast(mask, tf.float32))
        return loss




if __name__ == "__main__":
    embedding_dim = 256
    units = 512
    vocab_size = 5000
    max_length = 20

    m = Main(embedding_dim, units, vocab_size, max_length)
    csv_path = 'flickr30k_images/results.csv'  # Путь к CSV-файлу
    image_dir = 'flickr30k_images/flickr30k_images'   # Папка с изображениями
    m.main(csv_path, image_dir)