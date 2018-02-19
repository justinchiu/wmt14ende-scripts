ROOT=/n/rush_lab/data/wmt14-de-en
TEXT=${ROOT}/data/wmt14_en_de

DATA=$ROOT/data-onmt-master/wmt14_en_de.150-150.3-3

MODEL=/n/rush_lab/jc/onmt-master/wmt14-de-en/models
LOG=/n/rush_lab/jc/onmt-master/wmt14-de-en/logs
GEN=/n/rush_lab/jc/onmt-master/wmt14-de-en/gen

train_wmt14deen_master_baseline_brnn() {
    seed=$(od -A n -t d -N 1 /dev/urandom | tr -d ' ')
    name=baseline.brnn.s$seed
    mkdir -p $MODEL/$name
    mkdir -p $LOG
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $DATA \
        -save_model $MODEL/$name/$name \
        -seed $seed \
        -epochs 30 \
        -gpuid $1 \
        -batch_size 256 \
        | tee ${LOG}/$name.log
}

generate_wmtdeen_master_baseline_brnn() {
    mkdir -p $GEN
    name=baseline.brnn.s237
    python /n/home13/jchiu/python/OpenNMT-py/translate.py \
        -model $MODEL/$name/${name}_acc_63.81_ppl_6.37_e30.pt \
        -src $TEXT/test.de \
        -output $GEN/$name.test.en \
        -gpu $1 \
        -beam_size 5
}

train_wmt14deen_master_baseline_brnn() {
    seed=$(od -A n -t d -N 1 /dev/urandom | tr -d ' ')
    name=baseline.brnn.s$seed
    mkdir -p $MODEL/$name
    mkdir -p $LOG
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $DATA \
        -save_model $MODEL/$name/$name \
        -seed $seed \
        -epochs 30 \
        -gpuid $1 \
        -batch_size 256 \
        | tee ${LOG}/$name.log
}

generate_wmtdeen_master_baseline_brnn() {
    mkdir -p $GEN
    name=baseline.brnn.s237
    python /n/home13/jchiu/python/OpenNMT-py/translate.py \
        -model $MODEL/$name/${name}_acc_63.81_ppl_6.37_e30.pt \
        -src $TEXT/test.de \
        -output $GEN/$name.test.en \
        -gpu $1 \
        -beam_size 5
}
