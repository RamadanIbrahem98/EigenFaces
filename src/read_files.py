import os
import cv2
import matplotlib.pyplot as plt

class ImageReader:
  def __init__(self):
    self.__images = []
    self.__labels = []
    self.__images_paths = []
    self.__labels_paths = []
    self.__image_size = None
    self.__image_count = 0

  def __call__(self, dataset_path: str, mime_type: str) -> None:
      study = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
      for folder in study:
          for file in os.listdir(os.path.join(dataset_path, folder)):
              if file.endswith(mime_type):
                  self.__images_paths.append(os.path.join(dataset_path, folder, file))
                  self.__labels_paths.append(folder)
                  self.__image_count += 1
      return self.__load_images()

  def __load_images(self):
      sizes = {}
      temp_arr = []
      for image_path in self.__images_paths:
          temp_image = plt.imread(image_path)
          temp_arr.append(temp_image)
          sizes[temp_image.shape] = sizes.get(temp_image.shape, 0) + 1
      true_shape = max(sizes, key=sizes.get)

      self.__image_size = true_shape

      for i in range(len(temp_arr)):
          if temp_arr[i].shape[0] > true_shape[0] and temp_arr[i].shape[1] > true_shape[1]:
              temp_arr[i] = cv2.resize(temp_arr[i], dsize=(true_shape[1], true_shape[0]), interpolation=cv2.INTER_CUBIC)
          self.__images.append(temp_arr[i])
          self.__labels.append(self.__labels_paths[i])
      
      return self.__images

  @property
  def images(self) -> list:
    return self.__images

  @property
  def labels(self) -> list:
    return self.__labels

  @property
  def images_paths(self) -> list:
    return self.__images_paths

  @property
  def labels_paths(self) -> list:
    return self.__labels_paths

  @property
  def image_size(self) -> tuple:
    return self.__image_size

  @property
  def image_count(self) -> int:
    return self.__image_count
