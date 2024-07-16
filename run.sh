#!/bin/bash
function dataset() {
  input="/kaggle/input/kuaishou/KuaiSAR_final/src_inter.csv"
  output="/kaggle/working/rec_system/src/dataset"
  python src/dataset.py --input ${input} --output ${output}
}

dataset

#command=$1
#
#case $command in
#dataset)
#  dataset
#  ;;
#*)
#  echo "error command"
#  ;;
#esac
