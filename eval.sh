# DATA_ROOT=/data/****/VOC2012/VOC2012
# DATASET=voc
# TASK=15-1
DATA_ROOT=/data/****/ADEChallengeData2016/
DATASET=ade
TASK=100-10
EPOCH=50
BATCH=8
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=0
SUBPATH=BARM

CURR=5

python eval.py --data_root ${DATA_ROOT} --model deeplabv3clse_resnet101 --gpu_id 1 --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --w_transfer --amp --mem_size ${MEMORY} \
    --curr_step ${CURR} --subpath ${SUBPATH} 
