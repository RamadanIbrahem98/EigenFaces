import numpy as np

class AbstractDistance:
  def __init__(self, name):
    self.__name = name

  def __call__(self, p, q):
    raise NotImplementedError("Subclasses should implement this!")
  
  @property
  def name(self):
    return self.__name

class EuclideanDistance(AbstractDistance):
  def __init__(self):
    AbstractDistance.__init__(self,"EuclideanDistance")
  def __call__(self, p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sqrt(np.sum((p - q) ** 2))

class CosineDistance(AbstractDistance):
  def __init__(self):
    AbstractDistance.__init__(self,"CosineDistance")

  def __call__(self, p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return 1 - np.dot(p, q) / (np.sqrt(np.dot(p, p)) * np.sqrt(np.dot(q, q)))
