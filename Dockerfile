
FROM ubuntu:20.04

RUN apt-get update -y  &&  apt-get install -y  software-properties-common &&  add-apt-repository ppa:deadsnakes/ppa   && apt-get install  openjdk-8-jdk -y && apt-get  install python3-pip -y &&  export JAVA_HOME && apt-get  clean  && rm -rf  /var/lib/apt/lists/*

WORKDIR /qnai_model
COPY . /qnai_model
RUN pip3 install --no-cache-dir -r requirements.txt
ENV JAVA_HOME="/usr/lib/jvm/java-1.8-openjdk"
CMD ["python3", "./app.py"]