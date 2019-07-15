#!/bin/bash
#SBATCH --array=1-2
#SBATCH --job-name=orion_test
#SBATCH --output=other_logs/out_%a.log
#SBATCH --error=other_logs/err_%a.log
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=1Gb

IMAGEPATH = "path to image"
LABELPATH = "path to labels"

module load python/3.6
source ../../ve/bin/activate
orion -v hunt --config orion_config.yaml ../ecodse_funtime_alpha/main.py --config exp_config.yaml --log my_exp.log --imagepath $IMAGEPATH --labelpath $LABELPATH
