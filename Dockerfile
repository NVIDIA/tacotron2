FROM python:3.7

RUN apt-get update -y

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install jupyter

RUN pip install apex==0.9.10.dev0