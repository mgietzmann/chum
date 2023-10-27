FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential wget git vim curl

RUN apt-get -y install python3 python-is-python3
RUN apt-get -y install python3-pip

RUN pip install jupyterlab==4.0.7 \
                pandas==2.1.2 \
                plotly==5.18.0 \
                tqdm==4.66.1 \
                scipy==1.11.3 \
                scikit-learn==1.3.2
