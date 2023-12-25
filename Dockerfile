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

RUN apt-get -y install r-base r-base-dev
RUN apt-get -y install libxml2-dev libssl-dev pandoc
RUN apt-get -y install texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
RUN apt-get -y install libcurl4-gnutls-dev libxml2-dev libssl-dev libgit2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev 
RUN apt-get -y install libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev cmake libgdal-dev libudunits2-dev 
