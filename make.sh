set -e

export DOCKER_IMAGE=citibike-learn

if [[ "$1" = 'build' ]]; then
	echo 'Running docker build'
	docker build -f sagemaker/dockerfiles/Dockerfile_fresh -t ${DOCKER_IMAGE} .
	exit
fi


# require one arg.
if [[ $# != 2  || "$1" != 'push' ]]; then
	echo 'Syntax'
	echo 'make.sh build'
	echo 'make.sh push <image_tag>'
	exit
fi


if [[ "${AWS_ACCOUNT_ID}" = '' ]]; then
	echo missing AWS_ACCOUNT_ID var
	exit
fi

# export BIKELEARN_VERSION=0.3.12
DOCKER_TAG=$2
#export AWS_MODEL_NAME='pensive-swirles-2-12'

# echo "Building with BIKELEARN_VERSION=${BIKELEARN_VERSION}"
echo "Building with DOCKER_TAG=${DOCKER_TAG}"

# python setup.py sdist
#cp dist/bikelearn-${BIKELEARN_VERSION}.tar.gz sagemaker/mypackages
# cd sagemaker

# XXX former blah... ... also docker build
# echo 'Doing docker build'
# docker build  --build-arg BIKELEARN_VERSION=${BIKELEARN_VERSION}  -f dockerfiles/Dockerfile -t ${DOCKER_IMAGE} .
#

echo 'Doing docker tag'
docker tag ${DOCKER_IMAGE}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${DOCKER_IMAGE}:${DOCKER_TAG}

docker tag ${DOCKER_IMAGE}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${DOCKER_IMAGE}:latest
docker tag ${DOCKER_IMAGE}:latest ${DOCKER_IMAGE}:${DOCKER_TAG}

echo 'Doing login'
$(aws --profile adminuser ecr get-login --no-include-email --region us-east-1)

echo 'Doing docker push'
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${DOCKER_IMAGE}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${DOCKER_IMAGE}:${DOCKER_TAG}


# ... Run some integration tests too locally? 
# docker run -p 8080:8080 -v $(pwd)/local_test/test_dir:/opt/ml \
#    ${DOCKER_IMAGE}:${DOCKER_TAG} serve

# Make model
# python  update_client.py ${DOCKER_TAG} ${AWS_MODEL_NAME}

