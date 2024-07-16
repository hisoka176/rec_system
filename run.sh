#!/bin/bash
function dataset() {
  input="/kaggle/input/kuaishou/KuaiSAR_final/src_inter.csv"
  output="/kaggle/working/rec_system/src/dataset"
  python dataset.py --input ${input} --output ${output}
}

function train() {
  python train.py
}
command=$1
cd src
case $command in
'dataset')
  dataset
  ;;
'train')
  train
  ;;
*)
  echo "error command"
  ;;
esac
