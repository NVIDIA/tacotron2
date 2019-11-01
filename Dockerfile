FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update -y

RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorflow tensorboardX inflect==0.2.5 Unidecode==1.0.22 pillow jupyter

ADD apex /apex/
WORKDIR /apex/
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
