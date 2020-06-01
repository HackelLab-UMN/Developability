#!/bin/bash -l
#PBS -l walltime=24:00:00,nodes=1:ppn=24:gpus=2,mem=300gb -q v100 -A mart5523

module load conda
source activate awg-gpu

cd ./round3_gp2/Developability/
python3 main_seq_to_assay.py $PBS_ARRAYID
