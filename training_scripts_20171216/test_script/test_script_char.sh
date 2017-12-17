
NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/work/vincentzlt/tf-nmt/model-WMTsetting-20171216
MODEL=test_script_char
if [ ! -e ${MODEL_ROOT}/${MODEL} ]; then
mkdir ${MODEL_ROOT}/${MODEL}
fi

echo 'working on ' `pwd` 
echo 'save model to ' ${MODEL_ROOT}/${MODEL}


DATA_ROOT=`pwd`nmt/testdata/jc_mini_data/char
DATA_SPLIT=
SHARE_VOCAB=false
if [ ${SHARE_VOCAB} = true ]; then
    SHARE_VOCAB_DIR=share
else
    SHARE_VOCAB_DIR=noshare
fi


CUDA_VISIBLE_DEVICES=2 python3 -m nmt.nmt \
    --src=jp \
    --tgt=cn\
    --metrics=char_bleu,kytea_bleu \
    --text_format=char \
    --share_vocab=${SHARE_VOCAB} \
    --vocab_prefix=${DATA_ROOT}/${SHARE_VOCAB_DIR}/vocab \
    --train_prefix=${DATA_ROOT}/train \
    --dev_prefix=${DATA_ROOT}/dev \
    --test_prefix=${DATA_ROOT}/test \
    --out_dir=${MODEL_ROOT}/${MODEL} \
    --hparams_path=/clwork/vincentzlt/tf-nmt/nmt/standard_hparams/wmt16-batch64.json \
    --override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_`date "+%Y-%m-%d_%H_%M_%S"`

