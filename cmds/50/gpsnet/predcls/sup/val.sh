OUTPATH=$YOUR_PATH

#cd $SG
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10070 --nproc_per_node=1 \
        tools/relation_test_net.py --config-file "configs/sup-50.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR GPSNetPredictor \
        TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR $EXP/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH OUTPUT_DIR $OUTPATH   \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True
