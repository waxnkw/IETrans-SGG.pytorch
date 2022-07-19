OUTPATH=$EXP/50/transformer/predcls/lt/combine/rwt # replace with your own directory

#cd $SG
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --master_port 10076 --nproc_per_node=1 \
        tools/relation_test_net.py --config-file "configs/sup-50.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
        TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR $EXP/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH OUTPUT_DIR $OUTPATH   \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True
