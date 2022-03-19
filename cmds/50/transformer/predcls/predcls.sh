# train a supervised model
bash cmds/50/transformer/predcls/sup/train.sh
# conduct internal transfer
bash cmds/50/transformer/predcls/lt/internal/relabel.sh
# conduct external transfer
bash cmds/50/transformer/predcls/lt/external/relabel.sh
# combine internal and external transferred data
bash cmds/50/transformer/predcls/lt/combine/combine.sh
# train a new model
#bash cmds/50/transformer/predcls/lt/combine/train.sh
# train a new model using rwt
bash cmds/50/transformer/predcls/lt/combine/train_rwt.sh