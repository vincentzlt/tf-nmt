import os

nmt_root='/clwork/vincentzlt/tf-nmt/'
data_root=os.path.join(nmt_root,'data')
hparams_path=os.path.join(nmt_root,'nmt','standard_hparams','wmt16-batch64.json')


for src_lang in ['cn','jp']:
    for seg in ['char','comp','mecab-jieba','stroke','subword']:
        for is_share in [True, False]:
            # asign language
            if src_lang=='cn':
                tgt_lang='jp'
            else:
                tgt_lang='cn'
            cmd_ls.append('--src={}'.src_lang)
            cmd_ls.append('--tgt={}'.tgt_lang)
            # asign data dir
            if seg=='char':
                text_format='char'
                data_pref=os.path.join(data_root,seg)
                train_pref=os.path.join(data_pref,'train')
                dev_pref=os.path.join(data_pref,'dev')
                test_pref=os.path.join(data_pref,'test')
                vocab_pref=os.path.join(data_pref,'vocab.share') if is_share else os.path.join(data_pref,'vocab')

            elif seg=='mecab-jieba':
                text_format='char'
                data_pref=os.path.join(data_root,seg)
                train_pref=os.path.join(data_pref,'train')
                dev_pref=os.path.join(data_pref,'dev')
                test_pref=os.path.join(data_pref,'test')
                for vocab_size in ['16003','32003']:
                    vocab_pref=data_pref=os.path.join(data_root,'vocab.{}.share'.format(vocab_size)) if is_share else os.path.join(data_root,'vocab.{}'.format(vocab_size))





def gen_command(src,tgt,model,nmt_root,
                share_vocab,vocab_pref,
                train_pref,dev_pref,test_pref,
                out_dir,hpamars_path,text_format=None,subword_option=None):
    # cd to the nmt root
    cmd_ls=[]
    cmd_ls.append('cd {}'.format(nmt_root))
    # the basic cmd
    cmd_ls.append('CUDA_VISIBLE_DEVICES=0 python3 -m nmt.nmt')
    cmd_ls.append('--src={}'.format(src))
    cmd_ls.append('--tgt={}'.format(tgt))
    cmd_ls.append('--text_format={}'.format(text_format))
    if subword_option:
        cmd_ls.append('--subword_option={}'.format(subword_option))
    cmd_ls.append('--share_vocab={}'.format(str(share_vocab)))






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
