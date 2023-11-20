#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:~/framework

source /eos/home-k/kghorban/miniforge3/bin/activate framework_env
/eos/home-k/kghorban/miniforge3/envs/framework_env/bin/python ~/framework/run/decay_compare.py
