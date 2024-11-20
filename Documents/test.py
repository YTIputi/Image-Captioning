import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Убедитесь, что kaggle.json настроен
api = KaggleApi()
api.authenticate()

# Задайте путь к сохранению данных
dataset = 'hsankesara/flickr-image-dataset'
download_path = './data'

# Создайте папку для данных, если её нет
os.makedirs(download_path, exist_ok=True)

# Скачивание датасета
print("Downloading dataset...")
api.dataset_download_files(dataset, path=download_path, unzip=True)
print("Dataset downloaded and unzipped!")
