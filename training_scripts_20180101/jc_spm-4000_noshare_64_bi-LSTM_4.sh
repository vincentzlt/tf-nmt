NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/work/vincentzlt/tf-nmt/models-2018
MODEL=jc_spm-4000_noshare_64_bi-LSTM_4

echo 'working on ' $(pwd)
echo 'save model to ' ${MODEL_ROOT}/${MODEL}
mkdir ${MODEL_ROOT}/${MODEL} -p

DATA_ROOT=${NMT_ROOT}/data

CORPUS=subword # ('char','subword','mecab-jieba','comp','stroke')
SHARE_VOCAB=false
MODEL_TYPE=LSTM
SUBWORD_OPTION=
TEXT_FORMAT=
VOCAB_INFIX=


if [ ${CORPUS} = char ]; then
    DATA_PREFIX=${DATA_ROOT}/${CORPUS}


elif [ ${CORPUS} = subword ]; then
    SUBWORD_TYPE=spm        # ('bpe','spm')
    SUBWORD_VOCAB_SIZE=4000 # (4000 8000 16000 32000 64000)
    DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

    SUBWORD_OPTION=spm

elif [ ${CORPUS} = mecab-jieba ]; then
    DATA_PREFIX=${DATA_ROOT}/${CORPUS}
    VOCAB_INFIX=.16003

elif [ ${CORPUS} = comp ]; then
    SUBWORD_TYPE=spm        # ('bpe','spm')
    SUBWORD_VOCAB_SIZE=4000 # (1000 2000 4000)
    DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

    SUBWORD_OPTION=spm
    TEXT_FORMAT=comp

elif [ ${CORPUS} = stroke ]; then
    SUBWORD_TYPE=bpe        # ('bpe','spm')
    SUBWORD_VOCAB_SIZE=2000 # (1000 2000 4000)
    DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

    SUBWORD_OPTION=spm
    TEXT_FORMAT=stroke

fi

if [ ${SHARE_VOCAB} = false ]; then
    VOCAB_PREFIX=vocab${VOCAB_INFIX}
elif [ ${SHARE_VOCAB} = true ]; then
    VOCAB_PREFIX=vocab${VOCAB_INFIX}.share
fi
if [ ${MODEL_TYPE} = "LSTM" ];then
    hparam_path=${NMT_ROOT}/nmt/standard_hparams/lstm.json 
else
    hparam_path=${NMT_ROOT}/nmt/standard_hparams/gnmt.json
fi

CUDA_VISIBLE_DEVICES=2 python3 -m nmt.nmt \
    --src=jp \
    --tgt=cn \
    --metrics=char_bleu,kytea_bleu \
    --text_format=${TEXT_FORMAT} \
    --subword_option=${SUBWORD_OPTION} \
    --share_vocab=${SHARE_VOCAB} \
    --vocab_prefix=${DATA_PREFIX}/${VOCAB_PREFIX} \
    --train_prefix=${DATA_PREFIX}/train \
    --dev_prefix=${DATA_PREFIX}/dev \
    --test_prefix=${DATA_PREFIX}/test \
    --out_dir=${MODEL_ROOT}/${MODEL} \
    --hparams_path=${hparam_path} \
    --override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_$(date "+%Y-%m-%d_%H_%M_%S")
