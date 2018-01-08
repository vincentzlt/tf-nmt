MODEL=templet
CORPUS=comp # ('char','subword','mecab-jieba','comp','stroke','mecab-jieba_comp')
SUBWORD_OPTION=
TEXT_FORMAT=char
MODEL_ARCHITECTURE=LSTM
SHARE_VOCAB=false
RANDOM_SEED=12345
CUDA_VISIBLE_DEVICES=3
BATCH_SIZE=128
BATCH_SIZE_COEFFICIENT=3

NMT_ROOT=/clwork/vincentzlt/tf-nmt
cd ${NMT_ROOT}
MODEL_ROOT=/clwork/vincentzlt/tf-nmt/model-5_seeds

echo 'working on ' $(pwd)
echo 'save model to ' ${MODEL_ROOT}/${MODEL}
mkdir ${MODEL_ROOT}/${MODEL} -p

DATA_ROOT=/clwork/vincentzlt/tf-nmt/data_2

if [ ${CORPUS} = char ]; then
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}
	MAX_LEN=90

elif [ ${CORPUS} = subword ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=4000 # (4000 8000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}
	if [ ${SUBWORD_VOCAB_SIZE} = 4000 ]; then
		MAX_LEN=60
	elif [ ${SUBWORD_VOCAB_SIZE} = 8000 ]; then
		MAX_LEN=50
	fi
	SUBWORD_OPTION=spm

elif [ ${CORPUS} = mecab-jieba ]; then
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}
	MAX_LEN=52

elif [ ${CORPUS} = comp ]; then
	SUBWORD_TYPE=spm        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=8000 # (1000 2000 4000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	if [ ${SUBWORD_TYPE} = bpe ]; then
		if [ ${SUBWORD_VOCAB_SIZE} = 2000 ]; then
			MAX_LEN=90
		elif [ ${SUBWORD_VOCAB_SIZE} = 4000 ]; then
			MAX_LEN=70
		elif [ ${SUBWORD_VOCAB_SIZE} = 8000 ]; then
			MAX_LEN=62
		fi
	elif [ ${SUBWORD_TYPE} = spm ]; then
		if [ ${SUBWORD_VOCAB_SIZE} = 2000 ]; then
			MAX_LEN=88
		elif [ ${SUBWORD_VOCAB_SIZE} = 4000 ]; then
			MAX_LEN=73
		elif [ ${SUBWORD_VOCAB_SIZE} = 8000 ]; then
			MAX_LEN=66
		fi
	fi

	SUBWORD_OPTION=spm
	TEXT_FORMAT=comp

elif [ ${CORPUS} = stroke ]; then
	SUBWORD_TYPE=bpe        # ('bpe','spm')
	SUBWORD_VOCAB_SIZE=2000 # (1000 2000 4000)
	DATA_PREFIX=${DATA_ROOT}/${CORPUS}/${SUBWORD_TYPE}/${SUBWORD_VOCAB_SIZE}

	if [ ${SUBWORD_TYPE} = bpe ]; then
		if [ ${SUBWORD_VOCAB_SIZE} = 2000 ]; then
			MAX_LEN=100
		elif [ ${SUBWORD_VOCAB_SIZE} = 4000 ]; then
			MAX_LEN=83
		elif [ ${SUBWORD_VOCAB_SIZE} = 8000 ]; then
			MAX_LEN=76
		fi
	elif [ ${SUBWORD_TYPE} = spm ]; then
		if [ ${SUBWORD_VOCAB_SIZE} = 2000 ]; then
			MAX_LEN=93
		elif [ ${SUBWORD_VOCAB_SIZE} = 4000 ]; then
			MAX_LEN=83
		elif [ ${SUBWORD_VOCAB_SIZE} = 8000 ]; then
			MAX_LEN=79
		fi
	fi

	SUBWORD_OPTION=spm
	TEXT_FORMAT=stroke

fi

if [ ${SHARE_VOCAB} = false ]; then
	VOCAB_PREFIX=vocab
elif [ ${SHARE_VOCAB} = true ]; then
	VOCAB_PREFIX=vocab.share
fi

if [ ${MODEL_ARCHITECTURE} = LSTM ]; then
	HPARAM_PATH=${NMT_ROOT}/train_scripts_5_seeds/LSTM_max_len.json
elif [ ${MODEL_ARCHITECTURE} = GNMT ]; then
	HPARAM_PATH=${NMT_ROOT}/train_scripts_5_seeds/GNMT_max_len.json
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 -m nmt.nmt \
	--src=jp \
	--tgt=cn \
	--src_max_len=${MAX_LEN} \
	--tgt_max_len=${MAX_LEN} \
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
	--batch_size=$((${BATCH_SIZE} * ${BATCH_SIZE_COEFFICIENT})) \
	--num_train_steps=$((340000/${BATCH_SIZE_COEFFICIENT})) \
	--steps_per_stats=$((100/${BATCH_SIZE_COEFFICIENT})) \
	--override_loaded_hparams | tee ${MODEL_ROOT}/${MODEL}/log_$(date "+%Y-%m-%d_%H_%M_%S")
