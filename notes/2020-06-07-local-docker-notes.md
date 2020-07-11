

#### local development
* This repo has a `sagemaker/dockerfiles/Dockerfile_fresh` dockerfile for xgboost+jupyter portability
* Build a docker image locally, here calling it `citibike-learn`
```
docker build -f sagemaker/dockerfiles/Dockerfile_fresh -t citibike-learn .
```

* Run it, using my local data dir, `/Users/me/blah/blah/blah/data/citibike` where I have csv files kindly provided on the citibik data website.
```
my_local_data_directory=/Users/me/blah/blah/blah/data/citibike
docker run -p 8889:8889 -i -t -v $(pwd):/opt/program \
            -v ${my_local_data_directory}:/opt/data \
            citibike-learn:latest
```
* And since in the above command the `8889` port is exposed, then in the docker, we are able to run a jupyter server..
```
jupyter notebook --ip 0.0.0.0 --port 8889 --no-browser --allow-root
```
