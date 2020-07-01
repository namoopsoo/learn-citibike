import numpy as np

def create_random_dataset_known_proportion():
    array = np.array([1, 2, 3, 4, 5])
    size = array.shape[0]
    new_size = 10000
    newarray = np.random.choice(array, replace=True,
                    size=new_size, p=[.2, .2, .2, .2, .2])

    # print({k:v/new_size for (k,v) in dict(Counter(newarray)).items()})
    return newarray

