OUTPATH=$EXP/50/sgdet/sup/sup
mkdir -p $OUTPATH

#CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10081 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR GPSNetPredictor \
  SOLVER.PRE_VAL False \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 DTYPE "float32" \
  SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False   MODEL.ROI_HEADS.DETECTIONS_PER_IMG 64

