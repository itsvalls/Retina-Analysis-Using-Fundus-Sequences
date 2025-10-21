import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_retina_heatmap(predictions_csv, output_path):

    df = pd.read_csv(predictions_csv)

    
    df["Normal_Prob"] = 1 - df["Probability"]
    df["Diseased_Prob"] = df["Probability"]

    
    heatmap_data = df[["Normal_Prob", "Diseased_Prob"]].T
    heatmap_data.columns = df["Frame"]

    
    heatmap_np = heatmap_data.to_numpy()
    row_min = heatmap_np.min(axis=1)[:, np.newaxis]
    row_max = heatmap_np.max(axis=1)[:, np.newaxis]
    heatmap_norm = (heatmap_np - row_min) / (row_max - row_min + 1e-8)

    
    heatmap_data = pd.DataFrame(
        heatmap_norm, index=heatmap_data.index, columns=heatmap_data.columns
    )

    
    sns.set(context="paper", style="white")
    g = sns.clustermap(
        heatmap_data,
        cmap="RdBu_r",      
        linewidths=0.2,
        figsize=(12, 6),
        col_cluster=False,  
        row_cluster=True     
    )

    plt.suptitle("Retinal Frame-wise Disease Probability Heatmap", y=1.02, fontsize=14)

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f" Retina-style heatmap saved at {output_path}")


if __name__ == "__main__":
    generate_retina_heatmap(
        predictions_csv="data/predictions.csv",      
        output_path="outputs/retina_heatmap.png"     
    )
