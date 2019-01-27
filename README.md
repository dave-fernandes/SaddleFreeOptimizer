# SaddleFreeOptimizer
A second order optimizer for TensorFlow that uses the Saddle-Free method of Dauphin _et al_. (2014) with some modifications.

## Algorithm
The algorithm is described by [Dauphin _et al_. \(2014\)](https://arxiv.org/abs/1406.2572). The implementation here follows this paper with the following exceptions:
* The order of operations in the Lanczos method follows that recommended by [Paige \(1972\)](https://academic.oup.com/imamat/article-abstract/10/3/373/824284).
* The type of damping applied to the curvature matrix in the Krylov subspace has 3 options that can be specified in the optimizer's constructor.
* Instead of applying multiple damping coefficients and finding the result with the lowest loss, this implementation uses a Marquardt-style heuristic to update the damping coefficient as per [Martens \(2010\)](http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf).

## Files
* `SFOptimizer.py` is the optimizer class.
* `mnist/dataset.py` is a utility class from https://github.com/tensorflow/models.git used to obtain MNIST data.
* `XOR_Test.ipynb` is a Jupyter notebook containing a simple network trained to an XOR function.
* `AE_Test.ipynb` is a Jupyter notebook containing a deep autoencoder network trained with MNIST data.

## Implementation Notes
* The Lanczos iteration loop is unrolled into branches of the TensorFlow graph. This allows a full step to be taken in one TF operation. However, it means the graph can get large if you use a high Krylov dimension.
* As in the original paper, no re-orthogonalization is used for the Lanczos vectors. This means that they will likely become linearly dependent if the Krylov dimension is high \(> 70?\). There would, thus, be little benefit in attempting this.
