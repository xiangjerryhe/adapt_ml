from functools import wraps
from functools import partial
from abc import ABCMeta,abstractmethod
from importlib import import_module

class AdaptML(metaclass=ABCMeta):
  def __init__(self):
    self.np = self.import_numpy()
    self.fill_missing_functions()
  @staticmethod
  def bind(inst, func, new_func_name):
    return setattr(inst, new_func_name, func.__get__(inst, inst.__class__))
  
  def import_numpy(self):
    import numpy as np
    return np
  @abstractmethod
  def fill_missing_functions(self):
    pass

def argsort(np, seq):
  """takes 2 dimensional input, unlike in regular numpy which can handle 1D inputs too"""
  nrow, ncol = seq.shape
  return np.array([[i for (v, i) in sorted((v, i) for (i, v) in enumerate(row))] for row in seq])

class TensorflowNumpy(AdaptML):
  @staticmethod
  def import_numpy():
    import tensorflow.experimental.numpy as np
    return np
  def fill_missing_functions(self):
    setattr(self.np, "argsort", partial(argsort, self.np).__call__)

class RegularNumpy:
  def __init__(self):
    import numpy
    self.np = numpy
    
class JaxNumpy(AdaptML):
  @staticmethod
  def import_numpy():
    import jax.numpy as np
    return np
  def fill_missing_functions(self):
    pass
 
def ML(numpy):
  import numpy as np
  jax_numpy = JaxNumpy()
  tf_numpy = TensorflowNumpy()
  numpy_lookup = {"tf":tf_numpy,
                  "tensorflow":tf_numpy, 
                  "jax": jax_numpy,
                   "jax.numpy":jax_numpy,
                  "tensorflow.numpy":tf_numpy, 
                  "numpy":RegularNumpy()
                  }
  def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
          return fn(numpy_lookup[numpy], *args, **kwargs)
        return wrapper
  return decorate


@ML(numpy="tf")
def least_square(self, x, y):
    """ Computes the least-squares solution to a linear matrix equation. """
    np = self.np
    x_avg = np.average(x)
    y_avg = np.average(y)
    dx = x - x_avg
    dy = y - y_avg
    var_x = np.sum(dx**2)
    cov_xy = np.sum(dx * dy)
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)

if __name__=="__main__":
  # test least_square
  import numpy as np
  np.random.seed(0); N = 500000 ; X, Y = np.random.rand(N), np.random.rand(N)
  print(lstsqr(X, Y))
  # test argsort
  import numpy as np
  x = np.array([[3, 1, 2]])
  print(TensorflowNumpy().np.argsort(x))
