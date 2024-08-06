#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate emuflow
which python
echo $(for i in $(seq 1 50); do printf "-"; done)

echo Sampling Planck Lite with Cobaya
echo $(for i in $(seq 1 100); do printf "-"; done)
date_start=$(date +%s)
python sampleplanck.py nsamples=5 output_name=plancktest
date_end=$(date +%s)
seconds=$((date_end - date_start))
minutes=$((seconds / 60))
seconds=$((seconds - 60 * minutes))
hours=$((minutes / 60))
minutes=$((minutes - 60 * hours))
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo $(for i in $(seq 1 100); do printf "-"; done)

echo Sampling Planck Lite and DES with Cobaya
echo $(for i in $(seq 1 100); do printf "-"; done)
date_start=$(date +%s)
python sampledesplanck.py nsamples=5 output_name=desplancktest useflow=False
date_end=$(date +%s)
seconds=$((date_end - date_start))
minutes=$((seconds / 60))
seconds=$((seconds - 60 * minutes))
hours=$((minutes / 60))
minutes=$((minutes - 60 * hours))
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo $(for i in $(seq 1 100); do printf "-"; done)

echo Sampling Planck Lite and DES with Cobaya
echo $(for i in $(seq 1 100); do printf "-"; done)
date_start=$(date +%s)
python sampledesplanck.py nsamples=5 output_name=desplancktestflow useflow=True flow_name=base_plikHM_TTTEEE_lowl_lowE
date_end=$(date +%s)
seconds=$((date_end - date_start))
minutes=$((seconds / 60))
seconds=$((seconds - 60 * minutes))
hours=$((minutes / 60))
minutes=$((minutes - 60 * hours))
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo $(for i in $(seq 1 100); do printf "-"; done)

# addqueue -n 2x4 -s -q cmb -c tests -m 8 ./submit.sh
