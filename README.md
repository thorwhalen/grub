
# grub
A ridiculously simple search engine


# Example: Search code


```python
from grub.grub import SearchStore
```


```python
import sklearn  # instead of talking any file, let's search the files of sklearn itself!

path_format = os.path.dirname(sklearn.__file__) + '{}.py'
search = SearchStore(path_format)
```


```python
search('ANN')  # see that it guessed, pretty early, that I was talking about neural networks
```




    array(['sklearn/tree/_export.py', 'sklearn/linear_model/_least_angle.py',
           'sklearn/feature_selection/_base.py',
           'sklearn/feature_selection/tests/test_variance_threshold.py',
           'sklearn/neural_network/tests/test_stochastic_optimizers.py',
           'sklearn/neural_network/__init__.py',
           'sklearn/neural_network/_stochastic_optimizers.py',
           'sklearn/neural_network/_multilayer_perceptron.py',
           'sklearn/neural_network/rbm.py',
           'sklearn/neural_network/tests/test_rbm.py'], dtype='<U75')




```python
search('how to calibrate the estimates of my classifier')  # and yep... good keyword promisses: robust, calibration, feature selection and validation...
```




    array(['sklearn/covariance/_robust_covariance.py',
           'sklearn/svm/_classes.py',
           'sklearn/covariance/_elliptic_envelope.py',
           'sklearn/neighbors/_lof.py', 'sklearn/ensemble/_iforest.py',
           'sklearn/feature_selection/_rfe.py', 'sklearn/calibration.py',
           'sklearn/model_selection/_validation.py',
           'sklearn/ensemble/_forest.py', 'sklearn/ensemble/_gb.py'],
          dtype='<U75')


