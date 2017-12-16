
NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/work/vincentzlt/tf-nmt/model-WMTsetting-20171216
MODEL=cj_sp4k_noshare_aspec
if [ ! -e ${MODEL_ROOT}/${MODEL} ]; then
mkdir ${MODEL_ROOT}/${MODEL}
fi

echo 'working on ' `pwd` 
echo 'save model to ' ${MODEL_ROOT}/${MODEL}


DATA_ROOT=/work/vincentzlt/tf-nmt/data
DATA_SPLIT=sp_4k
SHARE_VOCAB=false
if [ ${SHARE_VOCAB} = true ]; then
    SHARE_VOCAB_DIR=share
else
    SHARE_VOCAB_DIR=noshare
fi


python -m nmt.nmt \
    --src=cn \
    --tgt=jp \
    --share_vocab=${SHARE_VOCAB} \
    --vocab_prefix=${DATA_ROOT}/${DATA_SPLIT}/${SHARE_VOCAB_DIR}/vocab \
    --train_prefix=${DATA_ROOT}/${DATA_SPLIT}/train \
    --dev_prefix=${DATA_ROOT}/${DATA_SPLIT}/dev \
    --test_prefix=${DATA_ROOT}/${DATA_SPLIT}/test \
    --out_dir=${MODEL_ROOT}/${MODEL} \
    --hparams_path=/clwork/vincentzlt/tf-nmt/nmt/standard_hparams/wmt16-batch64.json \    --override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_`date "+%Y-%m-%d_%H_%M_%S"`

