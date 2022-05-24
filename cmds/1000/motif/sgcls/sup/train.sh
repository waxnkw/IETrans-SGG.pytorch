OUTPATH=$EXP/1800/motif/sgcls/sup/sup
mkdir -p $OUTPATH

#CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=0,1,2,4 python -m torch.distributed.launch --master_port 10081 --nproc_per_node=4 \
  tools/relation_train_net.py --config-file "configs/sup-1000.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.PRE_VAL True \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 4 DTYPE "float32" \
  SOLVER.MAX_ITER 40000 SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 1000 \
  GLOVE_DIR $EXP/glove MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False
