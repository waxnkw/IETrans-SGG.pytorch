OUTPATH=$EXP/50/transformer/predcls/lt/combine/rwt
mkdir -p $OUTPATH
cp $EXP/50/transformer/predcls/lt/combine/relabel/em_E.pk $OUTPATH/em_E.pk

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --master_port 10097 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
  DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
  SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 16000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  TEST.INFERENCE "SOFTMAX"  IETRANS.RWT True \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  SOLVER.BASE_LR 0.005 \
  WSUPERVISE.LOSS_TYPE  ce_rwt  WSUPERVISE.DATASET InTransDataset  WSUPERVISE.SPECIFIED_DATA_FILE  $OUTPATH/em_E.pk