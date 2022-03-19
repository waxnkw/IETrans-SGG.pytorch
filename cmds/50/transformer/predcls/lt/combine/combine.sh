OUTPATH=$EXP/50/transformer/predcls/lt/combine/relabel
mkdir -p $OUTPATH
cd $OUTPATH
cp $SG/tools/ietrans/combine.py ./
python combine.py transformer
