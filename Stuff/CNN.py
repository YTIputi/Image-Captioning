import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class Test:
    def __init__(self, image_path: str):
        # Пример использования
        self.image_path = image_path  # Замените на путь к изображению example.jpg
        
    def main(self) -> None: 
        encoder = self.build_encoder()
        image = self.preprocess_image(self.image_path)
        features = encoder.predict(image)  # Извлекаем признаки изображения
        print(features.shape)  # Выводим форму признаков
    
    # Загружаем предобученную модель InceptionV3
    def build_encoder(self) -> tf.keras.Model:
        base_model = InceptionV3(weights='imagenet')  # Загружаем предобученную модель
        base_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Убираем классификационные слои
        return base_model

    # Функция для загрузки и предобработки изображения
    def preprocess_image(self, image_path: str) -> tf.Tensor:
        img = load_img(image_path, target_size=(299, 299))  # Изменяем размер изображения
        img = img_to_array(img)  # Преобразуем изображение в массив
        img = tf.keras.applications.inception_v3.preprocess_input(img)  # Подготавливаем изображение для InceptionV3
        img = tf.expand_dims(img, axis=0)  # Добавляем размерность батча
        return img


if __name__ == "__main__":
    t = Test('example.png')
    t.main()