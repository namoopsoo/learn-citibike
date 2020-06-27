
#### the two xgboost APIs
* [stack overflow q](https://stackoverflow.com/questions/50163393/xgboost-train-versus-xgbclassifier)


#### batch training
* [stack overflow](https://datascience.stackexchange.com/questions/47510/how-to-reach-continue-training-in-xgboost) on continuing training with xgboost fit

#### caching
* [external memory](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py)
* from the [large dataset](https://towardsdatascience.com/build-xgboost-lightgbm-models-on-large-datasets-what-are-the-possible-solutions-bf882da2c27d) perspective 

#### csv, libsvm
* [csv and libsvm utils](https://github.com/aws/sagemaker-xgboost-container/blob/master/src/sagemaker_xgboost_container/data_utils.py)
* [nice libsvm summary](https://stats.stackexchange.com/questions/61328/libsvm-data-format#65771)
* [csv to libsvm](https://stackoverflow.com/questions/24162544/change-training-data-to-libsvm-format-to-pass-it-to-grid-py-in-libsvm#24182510), copying the notes here:...

```
csv2libsvm.py <input file> <output file> [<label index = 0>] [<skip headers = 0>]
```
```
 python csv2libsvm.py mydata.csv libsvm.data 0 True
```
_"Convert CSV to LIBSVM format. If there are no labels in the input file, specify label index = -1. If there are headers in the input file, specify skip headers = 1."_

* [numpy and csv](https://stackoverflow.com/questions/24659814/how-to-write-a-numpy-array-to-a-csv-file)
* [write libsvm format](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.dump_svmlight_file.html)
