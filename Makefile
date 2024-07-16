.PHONY: dataset
dataset:
	filepath=/kaggle/input/kuaishou/KuaiSAR_final/src_inter.csv
	python src/dataset.py --file ${filepath}