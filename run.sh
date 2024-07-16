#!/bin/bash
function dataset() {
  input="/kaggle/input/kuaishou/KuaiSAR_final/src_inter.csv"
  output="/kaggle/working/rec_system/src/dataset"
  python src/dataset.py --input ${input} --output ${output}
}

function train() {
  python src/train.py
}
command=$1

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
