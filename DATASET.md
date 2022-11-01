# Download:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Download [image_data.json](https://thunlp.oss-cn-qingdao.aliyuncs.com/vg/image_data.json). Put the file at `datasets/vg`.
3. Download the VG-50 data from [GoogleDrive](https://drive.google.com/file/d/1JWa9DAxIlUc5wZsL6QM_29awKIGh7WrK/view?usp=sharing). Extract these files to `datasets/` and forms the file structure `datasets/vg/50`.
4. Download the VG-1800 data from [GoogleDrive](https://drive.google.com/file/d/1amdwuPcxZuXVU4W40FhzBi-AL1vbdCMJ/view?usp=sharing). Extract these files to `datasets/` and forms the file structure `datasets/vg/1000`.
