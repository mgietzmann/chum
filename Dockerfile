FROM ubuntu:22.04
MAINTAINER "Marcel Gietzmann-Sanders" "marcelsanders96@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential wget git vim curl

RUN apt-get -y install r-base r-base-dev
RUN apt-get -y install libxml2-dev libssl-dev pandoc

RUN apt-get -y install texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
