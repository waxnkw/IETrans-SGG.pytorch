## DEMO
To inference the model on custom images, we provide examples for both SGCLS and SGDET.

## Run SGDET
Please first download the pre-trained checkpoint from [MODEL_ZOO.md](MODEL_ZOO.md).
Then modify the **last_checkpoint** file in **visualization/demo/** to the path of your model.

For SGDET, to use VG50 trained models, you can download from [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans/transformer_sgdet_I70_E100.pth) and run
```sh
bash cmds/50/transformer/demo_sgdet.sh visualization/demo_imgs/
```
To use VG1800 trained models, you can download from [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans/vg1800_motif_sgcls_I90_E100.pth) and run:
```sh
bash cmds/1000/motif/demo_sgdet.sh visualization/demo_imgs/
```
Note that we only train a SGCLS model on VG1800, so the bounding boxes will be 
automatically generated from the pre-trained MaskRCNN.

The path **visualization/demo_imgs/** can be modified to the directory for your own images.

## Run SGCLS
You can also choose to **provide the bounding boxes by your self**.
to use VG50 trained models, you can download from [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans/transformer_sgdet_I70_E100.pth) and run
```sh
bash cmds/50/transformer/demo_sgcls.sh visualization/demo_imgs/
```

To use VG1800 trained models, you can download from [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans/vg1800_motif_sgcls_I90_E100.pth) and run:
```sh
bash cmds/1000/motif/demo_sgcls.sh visualization/demo_imgs/
```
We specify the path to your bounding boxes with **TEST.CUSTUM_BBOX_PATH**. You can modify it to your own file.
The format of the bounding box file should be saved in pickle format, and should look like:
```python
# please refer to visualization/demo/demo_bbox.pk
{
    image_name1: [
        np.ndarray([[x1, y1, x2, y2], ... ]),
        [label1, label2, ...]
    ],
}
```

## Visualize
The output will be saved in **visualization/demo/custom_prediction.pk**.
Please run **vis.ipynb** for visualization.