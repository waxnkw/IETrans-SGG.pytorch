OUTPATH=$EXP/50/vctree/predcls/sup/sup
mkdir -p $OUTPATH

#CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10081 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
  SOLVER.PRE_VAL False \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
  SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  TEST.METRIC "R"
