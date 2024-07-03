MODEL=deeplabv3bga_resnet101
DATA_ROOT=/data/****/VOC2012
DATASET=voc
TASK=15-1
EPOCH=20
BATCH=8
LOSS=bce_loss
LR=0.001
THRESH=0.7
MEMORY=0

SUBPATH=BARM

CURR=1

now=$(date +"%Y%m%d_%H%M%S")
result_dir=./checkpoints/${SUBPATH}/${TASK}/

if [ ! -d ${result_dir} ]; then
    mkdir -p ${result_dir}
fi

# python train.py --data_root ${DATA_ROOT} --model ${MODEL} --gpu_id 6,7 --crop_val --lr ${LR} \
#     --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
#     --dataset ${DATASET} --task ${TASK} --lr_policy poly \
#     --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
#     --w_transfer --amp --mem_size ${MEMORY} \
#     --curr_step ${CURR} --subpath ${SUBPATH} \
#     --overlap

CUDA_VISIBLE_DEVICES=2,3 \
torchrun --nproc_per_node=2 --master_port=24129 \
train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp\
    --curr_step ${CURR} --subpath ${SUBPATH} \
    --overlap \
    | tee ${result_dir}/train-$now.log