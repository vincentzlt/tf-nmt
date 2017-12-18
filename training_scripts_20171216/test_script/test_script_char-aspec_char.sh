
NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/work/vincentzlt/tf-nmt/model-WMTsetting-20171218
MODEL=test_script_aspec_char
if [ ! -e ${MODEL_ROOT}/${MODEL} ]; then
mkdir ${MODEL_ROOT}/${MODEL} -p
fi

echo 'working on ' `pwd` 
echo 'save model to ' ${MODEL_ROOT}/${MODEL}


DATA_ROOT=/work/vincentzlt/tf-nmt/data_2/
DATA_SPLIT=char
SHARE_VOCAB=false
if [ ${SHARE_VOCAB} = true ]; then
    SHARE_VOCAB_DIR=share
else
    SHARE_VOCAB_DIR=noshare
fi


CUDA_VISIBLE_DEVICES=5 python3 -m nmt.nmt  \
    --src=jp \
    --tgt=cn\
    --metrics=char_bleu,kytea_bleu \
    --text_format="" \
    --subword_option="" \
    --share_vocab=${SHARE_VOCAB} \
    --train_prefix=${DATA_ROOT}/${DATA_SPLIT}/train \
    --dev_prefix=${DATA_ROOT}/${DATA_SPLIT}/dev \
    --test_prefix=${DATA_ROOT}/${DATA_SPLIT}/test \
    --out_dir=${MODEL_ROOT}/${MODEL} \
    --hparams_path=/clwork/vincentzlt/tf-nmt/nmt/standard_hparams/wmt16-batch64.json \
    --override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_`date "+%Y-%m-%d_%H_%M_%S"`

