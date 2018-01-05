NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/work/vincentzlt/tf-nmt/model-big-vocab-comp-stroke
MODEL=jc_stroke_bpe8k_noshare

echo 'working on ' $(pwd)
echo 'save model to ' ${MODEL_ROOT}/${MODEL}
mkdir ${MODEL_ROOT}/${MODEL} -p

DATA_ROOT=/work/vincentzlt/tf-nmt/data_2

CORPUS=mecab-jieba_stroke # ('char','subword','mecab-jieba','comp','stroke','mecab-jieba_comp','mecab-jieba_stroke')
SUBWORD_OPTION=
TEXT_FORMAT=
if [ ${CORPUS} = char ]; then
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}

elif [ /${CORPUS} = subword ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=4000 # (4000 8000 16000 32000 64000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	SUBWORD_OPTION=spm

elif [ ${CORPUS} = mecab-jieba ]; then
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}

elif [ ${CORPUS} = comp ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=2000 # (1000 2000 4000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	SUBWORD_OPTION=spm
	TEXT_FORMAT=comp

elif [ ${CORPUS} = mecab-jieba_stroke ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=8000 # (8000, 16000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	SUBWORD_OPTION=spm
	TEXT_FORMAT=comp

elif [ ${CORPUS} = mecab-jieba_comp ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=8000 # (8000, 16000)
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

SHARE_VOCAB=false
if [ ${SHARE_VOCAB} = false ]; then
	VOCAB_PREFIX=vocab
elif [ ${SHARE_VOCAB} = true ]; then
	VOCAB_PREFIX=vocab.share
fi

CUDA_VISIBLE_DEVICES=1 python3 -m nmt.nmt \
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
	--hparams_path=${NMT_ROOT}/nmt/standard_hparams/wmt16-batch64.json \
	--override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_$(date "+%Y-%m-%d_%H_%M_%S")
