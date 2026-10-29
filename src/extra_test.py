import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


'''
    In order to run this code, simply in the terminal run "python extra_test.py" or "python3 extrs_test.py".
    Make sure that you are in the same folder as the file is located in.
    You need to ensure that the files in the FILES are in the same folder / directory before running.
'''

ARM = "kfold"
METRICS = ["concordance", "ibs", "calibration_gap"]
HIGHER_IS_BETTER = ["concordance"]
FILES = ["medical.csv", "financial-crisis.csv", "financial-calm.csv"]
SIGNIFICANCE_THRESHOLD = 0.05
df = pd.concat(pd.read_csv(f) for f in FILES)
df = df[df.arm == ARM]

summary = []

# iterate each dataset
for dataset in df.dataset.unique():
    # for each dataset iterate one metric at a time for each model
    for metric in METRICS:
        x = df[(df.dataset == dataset) & (df.metric == metric)]
        table = x.groupby(["fold", "model"]).value.mean().unstack("model")   # rows = folds

    
        model_columns = [table[col] for col in table.columns]
        stat, p_value = friedmanchisquare(*model_columns)
        
        ranks = table.rank(axis=1, ascending=(metric not in HIGHER_IS_BETTER)).mean()

        row = ranks.round(2).to_dict()
        row.update(dataset=dataset, metric=metric, p=round(p_value, 4))
        summary.append(row)

        if p_value < SIGNIFICANCE_THRESHOLD:
            print(f"\n{dataset} | {metric} | Nemenyi p-values")
            print(sp.posthoc_nemenyi_friedman(table).round(3))

print("\nMean ranks (1 = best), arm =", ARM)
summary_df = pd.DataFrame(summary).set_index(["dataset", "metric"]).to_string()
print(summary_df)