#!/usr/bin/zsh

source ~/.profile
export CUDA_VISIBLE_DEVICES=5

for f in *char* *CompSp4k* *MecabJieba*;do

    ./${f}
done
