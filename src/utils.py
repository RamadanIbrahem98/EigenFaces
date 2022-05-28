import numpy as np

def stack_images(images):
  return np.dstack(images).reshape(images[0].shape[0], len(images))

def zero_mean_images(stacked_images):
  stacked_images_mean = np.mean(stacked_images, axis=1).reshape(1, stacked_images.shape[0])
  return stacked_images - np.tile(stacked_images_mean.T, (1, stacked_images_mean.shape[0]))

def covariance_matrix(stacked_images_zero_mean):
  return (1 / stacked_images_zero_mean.shape[1]) * np.dot(stacked_images_zero_mean.T, stacked_images_zero_mean)

def eigenfaces(cov_matrix, stacked_images_zero_mean, num_components=0):
  eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
  eigenfaces = np.dot(stacked_images_zero_mean, eigen_vectors)

  for i in range(eigenfaces.shape[1]):
    eigenfaces[:,i] = eigenfaces[:,i]/np.linalg.norm(eigenfaces[:,i])

  idx = np.argsort(-eigen_values)

  eigen_values = eigen_values[idx]
  eigenfaces = eigenfaces[:,idx]

  eigen_values = eigen_values[0:num_components].copy()
  eigenfaces = eigenfaces[:,0:num_components].copy()

  return [eigen_values, eigenfaces]

def project(X, u, mu=None):
  if (mu is None):
    return np.dot(u.T, X)
  return np.dot(u.T, X - mu)

def reconstruct(W, u, mu=None):
  if (mu is None):
    return np.dot(u, W)
  return np.dot(u, W) + mu

def change_range(image, new_min, new_max):
  return (((new_max - new_min) * (image - np.min(image))) / (np.max(image) - np.min(image))) + new_min
