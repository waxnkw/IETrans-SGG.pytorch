# Tips
**1. The results in paper.**
- We further optimize the code. 
The overall performance (F@K) of VCTree, Transformer and GPS-Net will be higher than the results reported in the paper.

**2. What is the difference between your VG-50 train/val/test split with [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)?**
- We pre-separate 5,000 images for validation in advance and set the split flag to `1`, 
while [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) dynamically separate them in the data loading.
Please ensure to use the VG-SGG-with-attri.h5 we provide in [50.zip](https://drive.google.com/file/d/1JWa9DAxIlUc5wZsL6QM_29awKIGh7WrK/view?usp=sharing).
Otherwise, it will lead to errors in the data loading.

**3. Why is the performance of plain baselines (_e.g._, Motif, VCTree) slightly different from previous works?**
- There are mainly 3 reasons:
(1) First, we fix a bug in previous evaluation that is to remove duplicated triplets in evaluation stage.
For some models, fixing the bug will lead to a slight performance degeneration.
(2) Second, in the relabeling process, `RandomHorizontalFlip` and `RandomHorizontalFlip` augmentation should be closed down.
For simplicity, we just close them for all procedures.
Using them in training might improve the overall performance slightly.
(3) Third, we find that turning off frequency bias in the model training can help to slightly improve the mR@K and F@K.
Thus, we turn off frequency bias for all plain models. 

**4. Reweighting and resampling for VG-1800.**
- We find that the powerful reweighting and resampling can not work well under VG-1800 settings.
In the training set, the ratio for the frequent predicate and the infrequent ones is too significant (_e.g._, 273348:1).
In this condition, reweighting and resampling can easily lead to that the model focus too much 
on the big amount of noisy tail predicate classes, and the performance will be extremely bad.

