FILELIST=/home/zeng/workspace/tensorflow-workspace/intra_facenet.txt
#FILELIST=/home/zeng/workspace/tensorflow-workspace/intra_id256.txt
OUTPUT_NAME=jiafacenet_intra

#CHECKPOINT_PATH=/data/zeng/GFace8_ft_0.1/model.ckpt-49372
CHECKPOINT_PATH=/data/zeng/xxx/model-20170512-110547.ckpt-250000
#NETWORK_NAME=mobilenet_v2_075
#NETWORK_SCOPE_NAME=MobilenetV2
NETWORK_SCOPE_NAME=InceptionResnetV1
NETWORK_NAME=inception_resnet_v1
IMAGE_SIZE=160
CHANNEL=3
FEATURE_DIM=512
NUM_CLASSES=10572
#NUM_CLASSES=168925
EMBEDDING_NAME=PreLogitsFlatten



FILELIST_EXTRA=/home/zeng/workspace/tensorflow-workspace/extra_facenet.txt
OUTPUT_NAME_EXTRA=jiafacenet_extra

python feature_extraction.py \
  --f ${FILELIST} \
  --n ${NETWORK_NAME} \
  --sn ${NETWORK_SCOPE_NAME} \
  --s ${IMAGE_SIZE} \
  --c ${CHANNEL} \
  --d ${FEATURE_DIM} \
  --nc ${NUM_CLASSES} \
  --path ${CHECKPOINT_PATH} \
  --e ${EMBEDDING_NAME} \
  --o ${OUTPUT_NAME}


python feature_extraction.py \
  --f ${FILELIST_EXTRA} \
  --n ${NETWORK_NAME} \
  --sn ${NETWORK_SCOPE_NAME} \
  --s ${IMAGE_SIZE} \
  --c ${CHANNEL} \
  --d ${FEATURE_DIM} \
  --nc ${NUM_CLASSES} \
  --path ${CHECKPOINT_PATH} \
  --e ${EMBEDDING_NAME} \
  --o ${OUTPUT_NAME_EXTRA}

