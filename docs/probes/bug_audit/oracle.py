"""Independent live NumPy contracts used by the Rust audit reproducers."""
from pathlib import Path
import warnings
import numpy as np
warnings.simplefilter('ignore', RuntimeWarning)
print('NumPy oracle:', np.__version__)
assert np.cov([[1., 2., 3.]], rowvar=False).shape == (3, 3)
assert np.isnan(np.cov([[1., 2., 3.]], rowvar=False)).all()
assert np.cov(np.empty((1, 0)), rowvar=False).shape == (0, 0)
for function in [np.unwrap, np.trapezoid]:
    try:
        function(np.array(3.), axis=0)
    except (ValueError, IndexError):
        pass
    else:
        raise AssertionError(function.__name__ + ' accepted scalar axis')
np.testing.assert_array_equal(np.polyadd([1., 0.], [-1., 2.]), [0., 2.])
np.testing.assert_array_equal(np.polysub([1., 0.], [1., 2.]), [0., -2.])
expected = np.quantile(np.array([-(2**63), 2**63-1], dtype=np.int64), [0., .5, 1.])
np.testing.assert_array_equal(expected, [float(-(2**63)), float(2**63-1), float(2**63-1)])
values = [-np.inf, -1., -0., 0., 1., np.inf, np.nan]
actual = np.argsort(np.array([complex(r, i) for r in values for i in values]), kind='stable')
np.testing.assert_array_equal(actual, [0,1,2,3,4,5,7,8,9,10,11,12,14,21,15,22,16,17,23,24,18,25,19,26,28,29,30,31,32,33,35,36,37,38,39,40,6,13,20,27,34,41,42,43,44,45,46,47,48])
x = -4*.8 + 9*.2
np.testing.assert_array_equal(np.histogram([x], bins=5, range=(-4,9))[0], [1,0,0,0,0])
assert np.zeros((0, (2**63 - 1) // 8), dtype=np.float64).shape == (0, (2**63 - 1) // 8)
for bad_cols in [2**63 - 1, 2**63]:
    try:
        np.zeros((0, bad_cols), dtype=np.float64)
    except ValueError:
        pass
    else:
        raise AssertionError(f'zeros((0, {bad_cols})) unexpectedly accepted')
for name in ['badbool', 'badtuple', 'dupkey']:
    path = Path('target/bug-audit') / (name + '.npy')
    if path.exists():
        try:
            np.load(path)
        except ValueError:
            pass
        else:
            raise AssertionError(str(path) + ' unexpectedly accepted')
print('PASS live oracle contracts')
