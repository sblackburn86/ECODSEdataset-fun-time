#!/bin/bash
rm -rf mlruns
rm -rf my_working_dir
IMAGEPATH="../../rainforest/fixed-train-jpg/"
LABELPATH="../../rainforest/train_v3.csv"
orion -v hunt --config orion_config.yaml ../ecodse_funtime_alpha/main.py --config exp_config.yaml --log my_exp.log --imagepath $IMAGEPATH --labelpath $LABELPATH
