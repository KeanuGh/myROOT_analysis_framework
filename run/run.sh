#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/user/k/kghorban/framework

source /eos/home-k/kghorban/miniforge3/bin/activate framework_env
# /eos/home-k/kghorban/miniforge3/envs/framework_env/bin/python /afs/cern.ch/user/k/kghorban/framework/run/stack_prelim.py
/eos/home-k/kghorban/miniforge3/envs/framework_env/bin/python /afs/cern.ch/user/k/kghorban/framework/run/decay_compare.py
