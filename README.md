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

**Run CCSNet** 

To run ```CCSNet```, you would have to provide the project name and path of the code base, data, and output.

```
export PROJECT="Run_CCSNet/My_first_run" 
export CODE_BASE=path/to/ccsnet
export DATA_PATH=path/to/data_dir
export RESULT_DIR=path/to/output
python <ccsnet-application>
```

To make life easy, you can create a .env file like this.

```
# .env

CODE_BASE=${HOME}/path/to/ccsnet
DATA_PATH=${HOME}/path/to/data_dir
RESULT_DIR=${HOME}/path/to/output
```
Then you can run the training application like this.

```
PROJECT="Run_CCSNet/My_first_run" python path/to/ccsnet/apps/train/trainer/trainer.py -e $path/to/.env
```

To run all the ```CCSNet``` applications at once you can create a ```ccsnet.sh``` file like this.

```
# ccsnet.sh

PROJECT=Publication_Test/Test_001
CODE_BASE=/home/hongyin.chen/anti_gravity/CCSNet

# Data generation
# PROJECT=$PROJECT CODE_BASE=$CODE_BASE python $CODE_BASE/apps/datagen/collections/run_omicron.py -e $CODE_BASE/.env (This line need to run seperately)
PROJECT=$PROJECT CODE_BASE=$CODE_BASE python $CODE_BASE/apps/datagen/collections/get_strain.py -e $CODE_BASE/.env


# Run Training
PROJECT=$PROJECT CODE_BASE=$CODE_BASE python $CODE_BASE/apps/train/trainer/trainer.py -e $CODE_BASE/.env

# Run Test datagen and testing 
for i in {1..3}
do
    PROJECT=$PROJECT CODE_BASE=$CODE_BASE python $CODE_BASE/apps/datagen/test_data/background_sampler.py -e $CODE_BASE/.env -s $i
    PROJECT=$PROJECT CODE_BASE=$CODE_BASE python $CODE_BASE/apps/tests/test_ccsnet.py -e $CODE_BASE/.env -s $i -r run_01
done
```
and run ```bash ccsnet.sh``` to get a happy life.
