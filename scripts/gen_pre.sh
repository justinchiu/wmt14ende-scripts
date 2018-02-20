ROOT=/n/rush_lab/data/wmt14-de-en
TEXT=${ROOT}/data/wmt14_en_de/
BPE=${ROOT}/data/wmt14_en_de.bpe
PHRASE=${ROOT}/data/wmt14_en_de.phrase

generate_wmt() {
    # 69min real
    cd $ROOT/data
    bash prepare-wmt14en2de.sh
}

preprocess_wmt14deen_opennmt_thresh_len_master(){
    # 10min real
    cd $ROOT
    mkdir -p data-onmt-master
    python /n/home13/jchiu/python/OpenNMT-py/preprocess.py \
        -train_src ${TEXT}/train.de -train_tgt ${TEXT}/train.en \
        -valid_src ${TEXT}/valid.de -valid_tgt ${TEXT}/valid.en \
        -src_vocab_size 80000 -tgt_vocab_size 80000 \
        -src_words_min_frequency 3 -tgt_words_min_frequency 3 \
        -src_seq_length 150 -tgt_seq_length 150 \
        -save_data data-onmt-master/wmt14_de_en.150-150.3-3
}

preprocess_wmt14ende_opennmt_thresh_len_master(){
    # 10min real
    cd $ROOT
    mkdir -p data-onmt-master
    python /n/home13/jchiu/python/OpenNMT-py/preprocess.py \
        -train_src ${TEXT}/train.en -train_tgt ${TEXT}/train.de \
        -valid_src ${TEXT}/valid.en -valid_tgt ${TEXT}/valid.de \
        -src_vocab_size 80000 -tgt_vocab_size 80000 \
        -src_words_min_frequency 3 -tgt_words_min_frequency 3 \
        -src_seq_length 150 -tgt_seq_length 150 \
        -save_data data-onmt-master/wmt14_en_de.150-150.3-3
}

