from tensorflow.keras.preprocessing.text import Tokenizer
import DecoderWithAttention
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from DecoderWithAttention import DecoderWithAttention


class Main:
    def __init__(self, embedding_dim, units, vocab_size):
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_size = vocab_size
        self.decoder = DecoderWithAttention(embedding_dim, units, vocab_size)
        self.encoder = self.build_encoder()

    def main(self):
        # Инициализация токенизатора
        tokenizer = Tokenizer(num_words=self.vocab_size)
        tokenizer.word_index = {'<start>': 1, '<end>': 2, '<unk>': 3, 'a': 4, 'cat': 5, 'dog': 6}

        # Пример генерации текста
        image_path = 'example.png'  # Укажите путь к изображению
        image = self.preprocess_image(image_path)
        caption = self.generate_caption(image, self.encoder, self.decoder, tokenizer, max_length=20)

        print(f"Generated Caption: {caption}")


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


if __name__ == "__main__":
    m = Main(256, 512, 5000)
    m.main()