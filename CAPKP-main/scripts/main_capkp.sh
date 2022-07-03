#!/bin/bash

cd ..

# custom config
DATA=/pub/data/hujie/coop
TRAINER=CAPKP

DATASET=$1
CFG=$2  # config file
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
CSC=$5  # class-specific context (False or True)
KGCN=$6 # lambda
CARD=$7 # Number of the GPU card to be used
PRUNE=$8 # True means prune, False means not prune
NCLASS=$9 # The number of classes in the dataset

for SEED in 1
do
    DIR=/pub/data/hujie/coop/output/test/${DATASET}/${TRAINER}/${KGCN}_lambda/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=${CARD} python3 train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.CAPKP.N_CTX ${NCTX} \
        TRAINER.CAPKP.CSC ${CSC} \
        TRAINER.CAPKP.KGCN ${KGCN} \
        TRAINER.CAPKP.PRUNE ${PRUNE} \
        TRAINER.CAPKP.DATASET ${DATASET} \
        TRAINER.CAPKP.NCLASS ${NCLASS} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done