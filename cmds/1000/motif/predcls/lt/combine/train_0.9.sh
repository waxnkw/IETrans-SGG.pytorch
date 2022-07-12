OUTPATH=$EXP/1800/motif/predcls/lt/combine/0.9
mkdir -p $OUTPATH
cp $EXP/1800/motif/predcls/lt/combine/relabel/0.9/em_E.pk $OUTPATH/em_E.pk

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --master_port 10091 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/wsup-1000.yaml" \
  DATASETS.TRAIN \(\"1000DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 40000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 1808 \
  SOLVER.PRE_VAL False  \
  TEST.INFERENCE "SOFTMAX"  IETRANS.RWT False \
  WSUPERVISE.LOSS_TYPE  ce WSUPERVISE.DATASET InTransDataset  WSUPERVISE.SPECIFIED_DATA_FILE   $OUTPATH/em_E.pk

