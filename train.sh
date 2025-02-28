EXP_NAME="train_set_1"
DATASETS="train_set_1"
DATASETS_TEST="val_set_1"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST 