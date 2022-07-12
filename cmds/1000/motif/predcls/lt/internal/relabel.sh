OUTPATH=$EXP/1800/motif/predcls/lt/internal/relabel
mkdir -p $OUTPATH
cp $EXP/1800/motif/predcls/sup/sup/last_checkpoint $OUTPATH/last_checkpoint

CUDA_VISIBLE_DEVICES=2,4 python -m torch.distributed.launch \
  --master_port 10084 --nproc_per_node=2 \
  tools/internal_relabel.py --config-file "configs/wsup-1000.yaml"  \
  DATASETS.TRAIN \(\"1000DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  WSUPERVISE.DATASET InTransDataset  EM.MODE E  WSUPERVISE.SPECIFIED_DATA_FILE  datasets/vg/1000/vg_sup_data.pk


cd $OUTPATH
cp $SG/tools/ietrans/internal_cut.py ./
cp $SG/datasets/vg/1000/VG-dicts.json ./VG-SGG-dicts-with-attri.json
python internal_cut.py 0.9