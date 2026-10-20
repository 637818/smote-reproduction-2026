# SMOTE reproduction (Chawla et al., JAIR 2002, Table 3)

`SMOTE_reproduction.ipynb` reproduces the main experiment of the paper: AUC of a decision tree trained with plain majority
undersampling versus SMOTE plus undersampling, stratified 10 fold cross validation, on Pima, Phoneme, Oil and Mammography.
SMOTE and the undersampling rule are implemented from scratch and tested inside the notebook.

## Run
```
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook SMOTE_reproduction.ipynb     # then Run All
```
The notebook downloads the four csv files into `data/` by itself (internet needed). A full run takes about 8 to 11 minutes on one CPU core;
set `REPEATS = 1` in the parameter cell for a quick run. It also works in Google Colab.

## Parameters (second code cell)
k = 5 neighbours; SMOTE 50, 100, 200, 300, 400, 500 percent; majority undersampling none, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200,
300, 400, 500, 600, 700, 800, 1000, 2000 percent; stratified 10 fold CV, resampling on training folds only; 3 repeats (fold seeds 0, 1, 2);
classifier DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2) as a stand in for C4.5.
