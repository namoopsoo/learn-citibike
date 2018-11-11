FROM ubuntu:16.04




RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         #python \
         nginx \
         bzip2 \
         ca-certificates \
         vim \
    && rm -rf /var/lib/apt/lists/*


# 
# RUN apt-get -y install apt-transport-https iproute2
# iproute2 , includes the 'ss' utility , 
#        ss - another utility to investigate sockets
# netstat 



# note from https://docs.datascience.com/en/master/appendix-1/dockerfile-basics-and-best-practices-1.html , 

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
/bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/conda && \
rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install python=2.7 numpy scipy scikit-learn pandas py-xgboost-cpu
RUN conda install ipython
RUN pip install ipdb flask gevent gunicorn


COPY mypackages /opt/program/mypackages
WORKDIR /opt/program
RUN pip install mypackages/bikelearn-0.2.7.tar.gz
RUN pip install mypackages/funkybob-2018.5.1.tar.gz



# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
# RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
#     pip install numpy scipy scikit-learn pandas flask gevent gunicorn && \
#         (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
#         rm -rf /root/.cache




# ENTRYPOINT ["python", "k-means-algorithm.py"]


# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE


ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
# WORKDIR /opt/program



