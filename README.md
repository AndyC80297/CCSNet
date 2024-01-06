# CCSNet
Installation of CCSNet enviroments

For data genration:
  conda create -n datagen python=3.9 -y
  ...

For CCSNet training:
  pip install poetry==1.7.1
  cd CCSNet/apps/train
  poetry install
  cd CCSNet/libs/ccsnet
  poetry install
