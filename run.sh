#!/usr/bin/env bash
function dataset() {
  filepath=/kaggle/input/kuaishou/KuaiSAR_final/src_inter.csv
  python src/dataset.py --file ${filepath}
}

command=$1

case $command in
dataset)
  dataset
  ;;
*)
  echo "error command"
  ;;
esac
