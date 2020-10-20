

#### Extending notes on karea, what is the worst karea possible
* Earlier notes https://github.com/namoopsoo/learn-citibike/blob/master/notes/2020-07-04-aws.md
* I was wondering since I had also written out how to calculate the theoretical worst logloss, I should consider the worst karea too.


```python
import numpy as np

y_test = np.array([1,
                   2,
                   1,
                   0,
                   2])
# Worst possible, always splitting the predictions in the other classes
y_prob = np.array([[.5, 0, .5],
                  [.5, .5, 0],
                  [.5, 0, .5],
                  [0, .5, .5],
                  [.5, .5, 0],])

#
import fresh.metrics as fm
fm.kth_area(y_test, y_prob,
            num_classes=3)                  

```
