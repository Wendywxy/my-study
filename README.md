
<img src="misc/adast2.PNG"  style="width: 70%; height: 70%"/>



## Requirmenets:
- Python3.7
- Pytorch=='1.6'
- Numpy
- Sklearn
- Pandas
- openpyxl
- umap

## Prepare datasets
We used three public datasets in this study:
- [Sleep-EDF (EDF)]
- [SHHS dataset (S1, S2)](https://sleepdata.org/datasets/shhs)

Data of each domain should be split into train/validate/test splits.
The domains IDs should be (a, b, c, ...). 

For example, the data files of domain 'a' should be 
`train_a.pt`, `val_a.pt`, and `test_a.pt`, such that `train_a.pt` is a dictionary.

`train_a.pt = {"samples": x-data, "labels: y-labels}`, and similarly `val_a.pt`, and `test_a.pt`.

## Training model 
You can update different hyperparameters in the model by updating `config_files/config.py` file.

To train the model, use this command:
```
python train_CD.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 1 --device cuda --plot_umap False
python train_CD_onlyatt.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 1 --device cuda --plot_umap True
python train_CD_divsub_permute.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 1 --device cuda --plot_umap False
python train_CD_divsub_shuffle.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 1 --device cuda --plot_umap False
python train_CD_divsub_random.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 1 --device cuda --plot_umap True
python train_CD_multi_run.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 20 --device cuda --plot_umap True
python train_CD_b2a_bestresult.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 1 --device cuda --plot_umap True
python train_CD_amongsub_leaveonout.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 20 --device cuda --plot_umap True

python train_CD_onlyatt_loadpt.py --experiment_description differentBatchSizes --run_description bs_128 --num_runs 1 --device cuda --plot_umap True
```
## Results![img.png](img.png)
The results include the final classification report of the average performance and a seprate folder for each 
cross-domain scenario having its log file and its own classification report.

