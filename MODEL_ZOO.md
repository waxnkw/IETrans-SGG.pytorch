## MODEL_ZOO

## VG-50
### IETrans Data (Augmented Dataset)
The augmented dataset is generated with Neural Motif.
To simplify, **I** represents internal transfer and **E** represents external transfer.
Type | Link 
-- | -- 
**I**-10%   | [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/intrans-em_E.pk_topk_0.1) 
**I**-30%   | [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/intrans-em_E.pk_topk_0.3) 
**I**-70%   | [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/intrans-em_E.pk_topk_0.7) 
**I**-30% + **E**-100%   | [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans-em_E.pk_0.3) 
**I**-70% + **E**-100%  | [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans-em_E.pk_0.7) 

The **I**-10% and 30% is much similar to the original dataset's predicate distribution.  

### Neural Motif
Task | Link 
-- | -- 
PREDCLS   | [link](https://drive.google.com/file/d/10dMDvHPk8WmOaBL0I1LT9FwnM1COYviS/view?usp=sharing) 
SGCLS   | [link](https://drive.google.com/file/d/1Myb_JR9bm32PUNEPyVypWIpTPqCrS_uS/view?usp=sharing) 
SGDET   | [link](https://drive.google.com/file/d/16Rv0VwN4pg3h_LidB_zFPcdefAKiXfs7/view?usp=sharing) 

### Transformer
Task | Link 
-- | -- 
SGDET   | [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans/transformer_sgdet_I70_E100.pth) 

## VG-1800
### Neural Motif
Task | Link 
-- | -- 
SGCLS **I**-90% + **E**-100%   | [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/ietrans/vg1800_motif_sgcls_I90_E100.pth) 
