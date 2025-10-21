"""
DECISION TREE VISUALIZATION FOR FRAME CLASSIFICATION
Shows the decision-making process for each frame
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

def create_decision_tree_visualization(csv_path, output_path="outputs/decision_tree.png"):
    """
    Creates a visual decision tree showing classification of frames
    """
    
    # Load predictions
    df = pd.read_csv(csv_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle("Retinal Disease Detection - Decision Flow", 
                 fontsize=16, fontweight='bold')
    
    # ========== TOP: DECISION TREE DIAGRAM ==========
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Root node
    root_box = FancyBboxPatch((3.5, 8.5), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor='lightblue', 
                              linewidth=2)
    ax1.add_patch(root_box)
    ax1.text(5, 9, "Input Frame", ha='center', va='center', 
             fontsize=12, fontweight='bold')
    
    # Decision node
    decision_box = FancyBboxPatch((3.5, 6.5), 3, 1,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor='lightyellow',
                                  linewidth=2)
    ax1.add_patch(decision_box)
    ax1.text(5, 7, "EfficientNetB0\nProbability ≥ 0.5?", 
             ha='center', va='center', fontsize=10)
    
    # Arrow from root to decision
    arrow1 = FancyArrowPatch((5, 8.5), (5, 7.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax1.add_artist(arrow1)
    
    # Left branch (Normal)
    normal_box = FancyBboxPatch((0.5, 4.5), 2.5, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='green', facecolor='lightgreen',
                               linewidth=3)
    ax1.add_patch(normal_box)
    ax1.text(1.75, 5, "NORMAL", ha='center', va='center',
             fontsize=11, fontweight='bold', color='darkgreen')
    
    # Right branch (Diseased)
    diseased_box = FancyBboxPatch((7, 4.5), 2.5, 1,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='red', facecolor='lightcoral',
                                 linewidth=3)
    ax1.add_patch(diseased_box)
    ax1.text(8.25, 5, "DISEASED", ha='center', va='center',
             fontsize=11, fontweight='bold', color='darkred')
    
    # Arrows to branches
    arrow_no = FancyArrowPatch((4.2, 6.7), (2.5, 5.5),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='green')
    ax1.add_artist(arrow_no)
    ax1.text(3, 6.3, "NO", fontsize=9, color='green', fontweight='bold')
    
    arrow_yes = FancyArrowPatch((5.8, 6.7), (7.5, 5.5),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='red')
    ax1.add_artist(arrow_yes)
    ax1.text(7, 6.3, "YES", fontsize=9, color='red', fontweight='bold')
    
    # Add detailed outcomes
    # Normal outcome details
    normal_detail = FancyBboxPatch((0.2, 2.8), 3.1, 1.3,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='green', facecolor='white',
                                  linewidth=1, linestyle='--')
    ax1.add_patch(normal_detail)
    ax1.text(1.75, 3.8, "Low Risk", fontsize=9, ha='center', fontweight='bold')
    ax1.text(1.75, 3.45, "No immediate action", fontsize=8, ha='center')
    ax1.text(1.75, 3.1, "Regular monitoring", fontsize=8, ha='center')
    
    arrow_normal_detail = FancyArrowPatch((1.75, 4.5), (1.75, 4.1),
                                         arrowstyle='->', mutation_scale=15,
                                         linewidth=1.5, color='green',
                                         linestyle='--')
    ax1.add_artist(arrow_normal_detail)
    
    # Diseased outcome details
    diseased_detail = FancyBboxPatch((6.7, 2.8), 3.1, 1.3,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='red', facecolor='white',
                                    linewidth=1, linestyle='--')
    ax1.add_patch(diseased_detail)
    ax1.text(8.25, 3.8, "High Risk", fontsize=9, ha='center', fontweight='bold')
    ax1.text(8.25, 3.45, "Consult ophthalmologist", fontsize=8, ha='center')
    ax1.text(8.25, 3.1, "Further tests required", fontsize=8, ha='center')
    
    arrow_diseased_detail = FancyArrowPatch((8.25, 4.5), (8.25, 4.1),
                                           arrowstyle='->', mutation_scale=15,
                                           linewidth=1.5, color='red',
                                           linestyle='--')
    ax1.add_artist(arrow_diseased_detail)
    
    # Statistics summary at bottom
    total_frames = len(df)
    diseased_count = (df['Prediction'] == 'Diseased').sum()
    normal_count = total_frames - diseased_count
    
    stats_box = FancyBboxPatch((0.5, 0.3), 9, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='gray', facecolor='whitesmoke',
                              linewidth=2)
    ax1.add_patch(stats_box)
    
    ax1.text(5, 1.7, "ANALYSIS SUMMARY", ha='center', va='center',
             fontsize=11, fontweight='bold')
    ax1.text(2.5, 1.2, f"Total Frames: {total_frames}", ha='center', 
             fontsize=9)
    ax1.text(5, 1.2, f"Normal: {normal_count} ({normal_count/total_frames*100:.1f}%)",
             ha='center', fontsize=9, color='green', fontweight='bold')
    ax1.text(7.5, 1.2, f"Diseased: {diseased_count} ({diseased_count/total_frames*100:.1f}%)",
             ha='center', fontsize=9, color='red', fontweight='bold')
    
    avg_prob = df['Probability'].mean()
    ax1.text(5, 0.7, f"Avg Disease Probability: {avg_prob:.3f}", 
             ha='center', fontsize=9)
    
    # ========== BOTTOM: FRAME-BY-FRAME CLASSIFICATION ==========
    ax2.set_xlim(-0.5, len(df) - 0.5)
    ax2.set_ylim(-0.5, 1.5)
    
    # Plot each frame as a bar
    colors = ['red' if pred == 'Diseased' else 'green' 
              for pred in df['Prediction']]
    
    bars = ax2.bar(range(len(df)), df['Probability'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add threshold line
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
                label='Decision Threshold (0.5)')
    
    # Styling
    ax2.set_xlabel("Frame Number", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Disease Probability", fontsize=11, fontweight='bold')
    ax2.set_title("Frame-wise Classification Results", 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add frame labels (show every 5th frame to avoid crowding)
    tick_positions = range(0, len(df), max(1, len(df)//10))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([str(i) for i in tick_positions], rotation=45)
    
    # Add annotations for diseased frames
    diseased_frames = df[df['Prediction'] == 'Diseased']
    for idx, row in diseased_frames.iterrows():
        if row['Probability'] > 0.7:  # Only annotate high-confidence detections
            ax2.annotate(f"{row['Probability']:.2f}",
                        xy=(idx, row['Probability']),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='red', alpha=0.3))
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Decision tree visualization saved at {output_path}")
    plt.close()
    
    # Create detailed frame list
    create_frame_classification_list(df, "outputs/frame_classification.txt")


def create_frame_classification_list(df, output_path):
    """
    Creates a text file with detailed frame-by-frame classification
    """
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FRAME-BY-FRAME CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Frames Analyzed: {len(df)}\n")
        diseased_count = (df['Prediction'] == 'Diseased').sum()
        f.write(f"Diseased Frames: {diseased_count} ({diseased_count/len(df)*100:.1f}%)\n")
        f.write(f"Normal Frames: {len(df)-diseased_count} ({(len(df)-diseased_count)/len(df)*100:.1f}%)\n\n")
        
        f.write("-"*70 + "\n")
        f.write(f"{'Frame':<20} {'Prediction':<15} {'Probability':<15} {'Status'}\n")
        f.write("-"*70 + "\n")
        
        for idx, row in df.iterrows():
            status = "🔴 ALERT" if row['Prediction'] == 'Diseased' else "🟢 CLEAR"
            f.write(f"{row['Frame']:<20} {row['Prediction']:<15} "
                   f"{row['Probability']:<15.4f} {status}\n")
        
        f.write("-"*70 + "\n\n")
        
        # Add risk assessment
        f.write("RISK ASSESSMENT:\n")
        if diseased_count > len(df) * 0.5:
            f.write("⚠️  HIGH RISK: Majority of frames show disease indicators\n")
        elif diseased_count > len(df) * 0.2:
            f.write("⚠️  MODERATE RISK: Several frames show disease indicators\n")
        elif diseased_count > 0:
            f.write("⚠️  LOW-MODERATE RISK: Few frames show disease indicators\n")
        else:
            f.write("✅ LOW RISK: No disease indicators detected\n")
    
    print(f"✅ Frame classification list saved at {output_path}")


if __name__ == "__main__":
    create_decision_tree_visualization("data/predictions.csv")