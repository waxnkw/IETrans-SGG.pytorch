OUTPATH="$EXP/1800/motif/predcls/lt/combine/relabel/$1"
mkdir -p $OUTPATH
cd $OUTPATH
cp $SG/tools/vg1800/combine.py ./
python combine.py motif $1
