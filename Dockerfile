FROM pytorch/pytorch:0.4_cuda9_cudnn7
RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorflow tensorboardX inflect==0.2.5 Unidecode==1.0.22 jupyter
