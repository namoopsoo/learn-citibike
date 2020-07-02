import numpy as np

import fresh.utils as fu
import fresh.test.utils as ftu


def test_foo():
    newarray = ftu.make_skewed_array(skew=[.1, .1, .3, .4, .1])
    print('skew:', fu.get_proportions(newarray))
    print('classes:', list(sorted(set(newarray))))
    print('size before', newarray.shape)

    _, newy = fu.balance_dataset(newarray, newarray, shrinkage=.5)

    print(fu.get_proportions(newy))
    print('size after', newy.shape)

    # TODO ... some func asserts the new proportions are roughly balanced

