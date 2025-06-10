from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_DI_SPD_WD(groups, df, feature_name, ref_val, pred_col):
    
    results = {}
    
    for group in groups:
        df_temp = df[df[feature_name].isin([ref_val, group])].copy()
        df_temp[f"{feature_name}_bin"] = df_temp[feature_name].apply(lambda x: 1 if x == ref_val else 0)
        df_numeric = df_temp.select_dtypes(include=[np.number])
        
        for col in [pred_col, f"{feature_name}_bin"]:
            if col not in df_numeric.columns:
                df_numeric[col] = df_temp[col]

        df_numeric = df_numeric.dropna(subset=[pred_col, f"{feature_name}_bin"])

        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_numeric,
            label_names=[pred_col],
            protected_attribute_names=[f"{feature_name}_bin"]
        )

        metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{f"{feature_name}_bin": 1}],  
        unprivileged_groups=[{f"{feature_name}_bin": 0}] 
        )
        
        # Wasserstein Distance
        # group_0_preds = df_numeric[df_numeric[f"{feature_name}_bin"] == 0][pred_col]
        # group_1_preds = df_numeric[df_numeric[f"{feature_name}_bin"] == 1][pred_col]
        # wd = wasserstein_distance(group_0_preds, group_1_preds)
        
        results[group] = {
        "Disparate Impact": metric.disparate_impact(),
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        #"Wasserstein Distance": wd
        }
        
    return results

def plot_fairness_metrics(results, col_name, refer_class):
    results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': col_name})
    results_df = results_df.sort_values(by=col_name)

    # only the two metrics you want to plot
    metrics = ["Disparate Impact", "Statistical Parity Difference"]
    titles = [
        f"Disparate Impact (Compared to {refer_class})",
        f"Statistical Parity Difference (Compared to {refer_class})",
    ]

    # now create 2 subplots instead of 3
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(
            data=results_df,
            x=col_name,
            y=metric,
            hue=col_name,
            palette='crest',
            dodge=False,
            legend=False,
            ax=ax
        )

        # reference lines & y‐limits
        if metric == "Disparate Impact":
            ax.axhline(1.0, linestyle='--', color='black')
            ax.set_ylim(0, max(results_df[metric].max() * 1.2, 1.1))
        else:  # Statistical Parity Difference
            ax.axhline(0.0, linestyle='--', color='black')
            ax.set_ylim(-0.25, 0.25)

        # annotate bars
        for idx, row in results_df.iterrows():
            value = row[metric]
            y_max = ax.get_ylim()[1]
            y_position = min(value + 0.01, y_max * 0.95)
            ax.annotate(f"{value:.2f}", xy=(idx, y_position),
                        ha='center', va='bottom', fontsize=9)

        ax.set_title(titles[i])
        ax.set_ylabel(metric)
        ax.set_xlabel(col_name)

    plt.tight_layout()
    plt.show()

# def plot_fairness_metrics(results, col_name, refer_class):
#     results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': col_name})
#     results_df = results_df.sort_values(by=col_name)

#     metrics = ["Disparate Impact", "Statistical Parity Difference", "Wasserstein Distance"]
#     titles = [
#         f"Disparate Impact (Compared to {refer_class})",
#         f"Statistical Parity Difference (Compared to {refer_class})",
#         #f"Wasserstein Distance (Compared to {refer_class})"
#     ]

#     fig, axes = plt.subplots(1, 3, figsize=(14, 4))
#     plt.subplots_adjust(wspace=0.3)  # Add horizontal space between plots

#     for i, metric in enumerate(metrics):
#         ax = axes[i]
#         sns.barplot(data=results_df, x=col_name, y=metric, hue=col_name,
#                     palette='crest', dodge=False, legend=False, ax=ax)

#         # Reference lines
#         if metric == "Disparate Impact":
#             ax.axhline(1.0, linestyle='--', color='black')
#             ax.set_ylim(0, max(results_df[metric].max() * 1.2, 1.1))
#         elif metric == "Statistical Parity Difference":
#             ax.axhline(0.0, linestyle='--', color='black')
#             ax.set_ylim(-0.25, 0.25)
#         elif metric == "Wasserstein Distance":
#             ax.set_ylim(0, max(results_df[metric].max() * 1.2, 0.1))

#         # Annotations
#         for idx, row in results_df.iterrows():
#             value = row[metric]
#             y_max = ax.get_ylim()[1]
#             y_position = min(value + 0.01, y_max * 0.95)
#             ax.annotate(f"{value:.2f}", xy=(idx, y_position),
#                         ha='center', va='bottom', fontsize=9)

#         ax.set_title(titles[i])
#         ax.set_ylabel(metric)
#         ax.set_xlabel(col_name)

#     plt.tight_layout()
#     plt.show()
