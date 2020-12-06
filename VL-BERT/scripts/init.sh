#!/usr/bin/env bash
SELF=$(dirname "$(realpath $0)")
cd $SELF/..
echo "[init.sh] pwd: $(pwd)"

cd ./common/lib/roi_pooling/
echo "Clean UP"
rm -rf build *.so
echo "Bulid ROI Pooling"
python3 setup.py build_ext --inplace
# cd ../../../

#cd ./refcoco/data/datasets/refer/
#make
#cd ../../../../



