import numpy as np
from utils import *
from Distance import EuclideanDistance
class EigenFaces:
  def __init__(self, num_components=0):
    self.__train_set_images = []
    self.__train_set_labels = []
    self.__test_set_images = []
    self.__test_set_labels = []
    self.__num_components = num_components
    self.__W = None

  def __call__(self, images: list, labels: list, num_training: int=200) -> None:
    images = [image.flatten() for image in images]
    random_images_idx = sorted(list(np.random.choice(len(images) - 1, size=num_training, replace=False)), reverse=True)
    self.__train_set_images = [images.pop(i) for i in random_images_idx]
    self.__train_set_labels = [labels.pop(i) for i in random_images_idx]
    self.__test_set_images = images
    self.__test_set_labels = labels
    training_images_stack = stack_images(self.__train_set_images)
    self.__zero_means = zero_mean_images(training_images_stack)
    self.__cov_matrix = covariance_matrix(self.__zero_means)
    _, self.__eigen_faces = eigenfaces(self.__cov_matrix, self.__zero_means, num_components=self.__num_components)
    self.__W = []
    for i in range(training_images_stack.shape[1]):
      projection = project(training_images_stack[:, i], self.__eigen_faces)
      self.__W.append(projection.reshape((-1, 1)))

  def predict(self, image: np.ndarray, dist_metric=EuclideanDistance()) -> str:
    projected_image = project(image, self.__eigen_faces)
    min_dist = np.finfo('float').max
    min_label = -1
    for i, projection in enumerate(self.__W):
      dist = dist_metric(projection, projected_image)
      if dist < min_dist:
        min_dist = dist
        min_label = self.__train_set_labels[i]
    return min_label

  def save_model(self) -> None:
    np.save('model/zero_mean.npy', self.__zero_means)
    np.save('model/projections.npy', self.__W)
    np.save('model/labels.npy', self.__train_set_labels)
    np.save('model/eigen_faces.npy', self.__eigen_faces)

  def load_model(self) -> None:
    self.__zero_means = np.load('model/zero_mean.npy')
    self.__W = np.load('model/projections.npy')
    self.__train_set_labels = np.load('model/labels.npy')
    self.__eigen_faces = np.load('model/eigen_faces.npy')

  @property
  def train_set_images(self) -> list:
    return self.__train_set_images
  
  @property
  def train_set_labels(self) -> list:
    return self.__train_set_labels

  @property
  def test_set_images(self) -> list:
    return self.__test_set_images

  @property
  def test_set_labels(self) -> list:
    return self.__test_set_labels

  @property
  def num_components(self) -> int:
    return self.__num_components

  @property
  def W(self) -> np.ndarray:
    return self.__W
