import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_confidence(csv_path, output_path="outputs/confidence_graph.png"):
    
    df = pd.read_csv(csv_path)

    
    expected_columns = ["Frame", "Probability"]
    if not all(col in df.columns for col in expected_columns):
        print(f" Expected columns {expected_columns}, but got: {df.columns.tolist()}")
        return

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["Frame"], df["Probability"], marker='o', linestyle='-', color='teal', label='Disease Probability')
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.xticks(rotation=90)
    plt.title("Model Confidence Over Time (Frame-wise)")
    plt.xlabel("Frame Name")
    plt.ylabel("Probability of Disease")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f" Graph saved at: {output_path}")


if __name__ == "__main__":
    plot_confidence("data/predictions.csv")
