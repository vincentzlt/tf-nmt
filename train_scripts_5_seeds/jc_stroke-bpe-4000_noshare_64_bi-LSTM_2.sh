MODEL=jc_stroke-bpe-4000_noshare_64_bi-LSTM_2
CORPUS=stroke # ('char','subword','mecab-jieba','comp','stroke','mecab-jieba_comp')
SUBWORD_OPTION=
TEXT_FORMAT=char
MODEL_ARCHITECTURE=LSTM
SHARE_VOCAB=false
RANDOM_SEED=22908
CUDA_VISIBLE_DEVICES=2

NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/clwork/vincentzlt/tf-nmt/model-5_seeds

echo 'working on ' $(pwd)
echo 'save model to ' ${MODEL_ROOT}/${MODEL}
mkdir ${MODEL_ROOT}/${MODEL} -p

DATA_ROOT=/clwork/vincentzlt/tf-nmt/data_2

if [ ${CORPUS} = char ]; then
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}

elif [ ${CORPUS} = subword ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=4000 # (4000 8000 16000 32000 64000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	SUBWORD_OPTION=spm

elif [ ${CORPUS} = mecab-jieba ]; then
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}

elif [ ${CORPUS} = comp ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=4000 # (1000 2000 4000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	SUBWORD_OPTION=spm
	TEXT_FORMAT=comp

elif [ ${CORPUS} = stroke ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=4000 # (1000 2000 4000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	SUBWORD_OPTION=spm
	TEXT_FORMAT=stroke

fi

if [ ${SHARE_VOCAB} = false ]; then
	VOCAB_PREFIX=vocab
elif [ ${SHARE_VOCAB} = true ]; then
	VOCAB_PREFIX=vocab.share
fi

if [ ${MODEL_ARCHITECTURE} = LSTM ]; then
	HPARAM_PATH=${NMT_ROOT}/train_scripts_5_seeds/LSTM.json
elif [ ${MODEL_ARCHITECTURE} = GNMT ]; then
	HPARAM_PATH=${NMT_ROOT}/train_scripts_5_seeds/GNMT.json
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 -m nmt.nmt \
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
	--hparams_path=${HPARAM_PATH} \
	--random_seed=${RANDOM_SEED} \
	--override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_$(date "+%Y-%m-%d_%H_%M_%S")
