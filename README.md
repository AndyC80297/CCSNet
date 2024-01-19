# CCSNet

## Installation of CCSNet environments


**Data generation for CCSNet**
```
  conda create -n datagen python=3.9 -y
  ...
```


**For CCSNet model training**
```
  pip install poetry==1.7.1
  cd CCSNet/apps/train
  poetry install
  cd CCSNet/libs/ccsnet
  poetry install
```
