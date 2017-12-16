
NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/work/vincentzlt/tf-nmt/model-WMTsetting-20171216
MODEL=test_script
if [ ! -e ${MODEL_ROOT}/${MODEL} ]; then
mkdir ${MODEL_ROOT}/${MODEL}
fi

echo 'working on ' `pwd` 
echo 'save model to ' ${MODEL_ROOT}/${MODEL}


DATA_ROOT=`pwd`/nmt/testdata/reverse-order
DATA_SPLIT=
SHARE_VOCAB=false
if [ ${SHARE_VOCAB} = true ]; then
    SHARE_VOCAB_DIR=share
else
    SHARE_VOCAB_DIR=noshare
fi


CUDA_VISIBLE_DEVICES=1 python -m nmt.nmt \
    --src=source \
    --tgt=target \
    --share_vocab=${SHARE_VOCAB} \
    --vocab_prefix=${DATA_ROOT}/train/vocab \
    --train_prefix=${DATA_ROOT}/train/data \
    --dev_prefix=${DATA_ROOT}/dev/data \
    --test_prefix=${DATA_ROOT}/test/data \
    --out_dir=${MODEL_ROOT}/${MODEL} \
    --hparams_path=/clwork/vincentzlt/tf-nmt/nmt/standard_hparams/wmt16-batch64.json \
    --override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_`date "+%Y-%m-%d_%H_%M_%S"`

