# nonholo_onlineshield

## Manage env

Go to MADDPG and MPE's root direcotry and run

`pip install -e .`

Ensure that `mpe` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`)

Make sure tensorflow's version

`pip install tensorflow==1.8.0`

and openai gym

`pip install gym==0.10.5`

## How to use the code

go to

`cd MADDPG/experiments`

start to train in the beginning

`python train.py --scenario test_env --save-dir ./tmp/policy/model1`

display trained model

`python train.py --scenario test_env --load-dir ./tmp/policy/model1 --display`

continue to train the model

`python train.py --scenario test_env --save-dir ./tmp/policy/model1 --restore`


