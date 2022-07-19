# train a supervised model
bash cmds/1000/motif/predcls/sup/train.sh
# conduct internal transfer
bash cmds/1000/motif/predcls/lt/internal/relabel.sh
# conduct external transfer
bash cmds/1000/motif/predcls/lt/external/relabel.sh
# combine internal and external transferred data
bash cmds/1000/motif/predcls/lt/combine/combine.sh
# train a new model
#bash cmds/1000/motif/predcls/lt/combine/train.sh
# train a new model using rwt
bash cmds/1000/motif/predcls/lt/combine/train_0.9.sh