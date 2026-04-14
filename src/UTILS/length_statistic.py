import os, sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def load_and_prepare_data(result_data_path,min_len:int=15,max_len:int=1024,round_k=10,random_status:int=None):
    df = pd.read_csv(result_data_path)
    df['length_diff'] = df['target_len'] - df['generate_len'] 
    df['MAE'] = df['length_diff'].abs()
    df['MSE'] = df['length_diff'] ** 2
    df[f"tg_len_round_{round_k}"] =  df["target_len"].apply(lambda x: round_k *(x//round_k)) #round at the nearest down round_k factor
    
    df['outlier_state'] = (df['MAE'] > 20).astype(int)

    if random_status is not None:
        mask_target_len = (df[f"tg_len_round_{round_k}"]<max_len) & (df[f"tg_len_round_{round_k}"]>min_len) & (df["random_status"]==random_status)
    else:
        mask_target_len = (df[f"tg_len_round_{round_k}"]<max_len) & (df[f"tg_len_round_{round_k}"]>min_len) 

    df = df.loc[mask_target_len]
    return df

def describe_and_save(df, path_to_save, columns):
    # print(df.shape)
    # print(df.describe().round(3))
    df_desc = df.describe().round(3)[columns].iloc[1:] #'bleu_score'
    df_desc.to_csv(Path(path_to_save, "df_describe.csv"))



def plot_metrics(df, path_to_save,round_k,model_name, metric_titles):


    for metric, title in metric_titles.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=df,
            x=f"tg_len_round_{round_k}",
            y=metric,
            color=".85", 
            linecolor="#137",
            showmeans=True,
            showfliers=True,
            meanprops={
                "marker": "o",              # circle marker
                "markerfacecolor": "#137", # fill color
                "markeredgecolor": "black", # border color
                "markersize": 4             # smaller size (default ~8)
            },
            ax=ax
        )

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel(f"Target length (binned by {round_k})")
        ax.set_ylabel(f"{title}")
        ax.grid(True, color='#d3d3d3', linestyle='--', linewidth=0.5)
        # Title
        ax.set_title(f"{title} distribution by target length\n{model_name}", fontsize=14, fontweight="bold")


        fig.tight_layout()
        fig.savefig(Path(path_to_save, f"boxplot_{title}.png"))
        # plt.show()
        plt.close(fig)


def plot_box_plot_length(df, path_to_save,round_k,model_name,showfliers=True):

    # Ordre catégoriel & positions
    order = sorted(df[f"tg_len_round_{round_k}"].unique())
    pos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(
        data=df, 
        x=f"tg_len_round_{round_k}", 
        y="generate_len", 
        order=order,
        showmeans=True, 
        showfliers=showfliers, 
        patch_artist=True,
        color=".85", linecolor="#137",
        medianprops=dict(color="green", linewidth=1),              # médiane verte (ligne épaisse)
        meanprops=dict(marker="v", markerfacecolor="red", markeredgecolor="black", markersize=6),     # moyenne: triangle rouge
        flierprops=dict(marker="o", markersize=5, markerfacecolor="white", markeredgecolor="black", alpha=0.7),       # outliers
        whiskerprops=dict(color="#0847ac"),                        # moustaches bleu
        capprops=dict(color="#0847ac"),
        ax=ax
    )

    ax.plot(pos, np.array(order)+ round_k/2 , linestyle=":", color="black", alpha=0.6)

    # Axes & titres
    ax.set_xticks(pos)
    ax.set_xticklabels(order, rotation=45)
    ax.set_xlabel("Target Length")
    ax.set_ylabel("Generated Length")
    ax.set_title(f"{model_name}")

    # --- Légende cohérente avec les styles ci-dessus ---
    if showfliers:
        legend_elements = [
            mlines.Line2D([], [], color='black', linestyle='--', label='Expected target length'),
            mlines.Line2D([], [], marker='v', linestyle='None', markerfacecolor='red', markeredgecolor='black', markersize=7, label='Mean'),
            mlines.Line2D([], [], color='green', linewidth=2, label='Median'),
            mlines.Line2D([], [], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', markersize=6, label='Outliers'),
        ]
    else:
        legend_elements = [
            mlines.Line2D([], [], color='black', linestyle='--', label='Expected target length'),
            mlines.Line2D([], [], marker='v', linestyle='None', markerfacecolor='red', markeredgecolor='black', markersize=7, label='Mean'),
            mlines.Line2D([], [], color='green', linewidth=2, label='Median'),
        ]

    ax.legend(handles=legend_elements, loc="upper left")

    ax.grid(True, color='#d3d3d3', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    # plt.show()

    fig.savefig(Path(path_to_save, f"boxplot_generated_over_expected.png"))



def plot_histogram(df, path_to_save, model_name, bins:list=[100, 100]):
    # Create two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Histogram for target length
    sns.histplot(
        data=df, x="target_len", bins=bins[0], kde=True, stat="density",
        ax=axes[0], color="skyblue"
    )
    axes[0].set_title('Target token Length')
    axes[0].set_xlabel('Number of tokens')
    axes[0].set_ylabel('Density')

    # Histogram for generated length
    sns.histplot(
        data=df, x="generate_len", bins=bins[1], kde=True, stat="density",
        ax=axes[1], color="salmon"
    )
    axes[1].set_title(f'{model_name} generated token tength')
    axes[1].set_xlabel('Number of tokens')
    axes[1].set_ylabel('')

    # Adjust layout and save
    fig.suptitle('Histogram of Token Count (Target vs Generated)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(path_to_save, "histplot_length.png"))
    # plt.show()
    plt.close(fig)



def main(path_metrics, path_to_save, round_k, model_name, columns:list, metric_titles:dict, min_len:int=15, max_len:int=900, showfliers:bool=True, random_status:int=None):

    try:
        df = load_and_prepare_data(path_metrics ,min_len , max_len,round_k=round_k, random_status=random_status)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    
    describe_and_save(df, path_to_save, columns)

    df = df[df["outlier_state"]==0]

    plot_metrics(df, path_to_save,round_k,model_name, metric_titles)

    plot_box_plot_length(df, path_to_save,round_k,model_name,showfliers)

    plot_histogram(df, path_to_save, model_name)
    
