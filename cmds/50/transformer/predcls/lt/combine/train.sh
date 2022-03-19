OUTPATH=$EXP/50/predcls/lt/combine/combine
mkdir -p $OUTPATH
cp $EXP/50/predcls/lt/combine/relabel/em_E.pk $OUTPATH/em_E.pk

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --master_port 10091 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
  DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  TEST.INFERENCE "SOFTMAX" \
  WSUPERVISE.LOSS_TYPE  softce WSUPERVISE.DATASET HIntraDataset  WSUPERVISE.CLIP_FILE  $OUTPATH/em_E.pk
