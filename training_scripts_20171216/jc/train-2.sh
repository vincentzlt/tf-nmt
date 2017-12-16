#!/usr/bin/zsh

source ~/.profile
export CUDA_VISIBLE_DEVICES=3

for f in *sp4k* *StrokeSp4k*;do

    ./${f}
done
