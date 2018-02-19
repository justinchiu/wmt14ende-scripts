ROOT=/n/rush_lab/data/wmt14-de-en
TEXT=${ROOT}/data/wmt14_en_de

DEENDATA=$ROOT/data-onmt-master/wmt14_de_en.150-150.3-3
ENDEDATA=$ROOT/data-onmt-master/wmt14_en_de.150-150.3-3

DEENMODEL=/n/rush_lab/jc/onmt-master/wmt14-de-en/models
DEENLOG=/n/rush_lab/jc/onmt-master/wmt14-de-en/logs
DEENGEN=/n/rush_lab/jc/onmt-master/wmt14-de-en/gen

ENDEMODEL=/n/rush_lab/jc/onmt-master/wmt14-en-de/models
ENDELOG=/n/rush_lab/jc/onmt-master/wmt14-en-de/logs
ENDEGEN=/n/rush_lab/jc/onmt-master/wmt14-en-de/gen

train_wmt14deen_master_baseline_brnn() {
    seed=$(od -A n -t d -N 1 /dev/urandom | tr -d ' ')
    name=de2en.baseline.brnn.s$seed
    mkdir -p $DEENMODEL/$name
    mkdir -p $DEENLOG
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $DEENDATA \
        -save_model $DEENMODEL/$name/$name \
        -seed $seed \
        -epochs 30 \
        -gpuid $1 \
        -batch_size 256 \
        | tee ${DEENLOG}/$name.log
}

generate_wmt14deen_master_baseline_brnn() {
    # old naming convetion, but it's fine.
    mkdir -p $DEENGEN
    name=baseline.brnn.s237
    python /n/home13/jchiu/python/OpenNMT-py/translate.py \
        -model $DEENMODEL/$name/${name}_acc_63.81_ppl_6.37_e30.pt \
        -src $TEXT/test.de \
        -output $DEENGEN/$name.test.en \
        -gpu $1 \
        -beam_size 5
}

train_wmt14ende_master_baseline_brnn() {
    seed=$(od -A n -t d -N 1 /dev/urandom | tr -d ' ')
    name=en2de.baseline.brnn.s$seed
    mkdir -p $ENDEMODEL/$name
    mkdir -p $ENDELOG
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $ENDEDATA \
        -save_model $ENDEMODEL/$name/$name \
        -seed $seed \
        -epochs 30 \
        -gpuid $1 \
        -batch_size 256 \
        | tee ${ENDELOG}/$name.log
}

generate_wmt14ende_master_baseline_brnn() {
    mkdir -p $ENDEGEN
    name=en2de.baseline.brnn.s237
    python /n/home13/jchiu/python/OpenNMT-py/translate.py \
        -model $ENDEMODEL/$name/${name}_acc_63.81_ppl_6.37_e30.pt.nonexistent \
        -src $TEXT/test.de \
        -output $ENDEGEN/$name.test.en \
        -gpu $1 \
        -beam_size 5
}
