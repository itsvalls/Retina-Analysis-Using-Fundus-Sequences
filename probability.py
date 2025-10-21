

import pandas as pd
import numpy as np
import os

def explain_probability_calculation(predictions_csv="data/predictions.csv"):
    """
    Creates a detailed CSV showing probability calculation breakdown
    """
    
   
    df = pd.read_csv(predictions_csv)
    
    
    calculation_df = df.copy()
    
  
    calculation_df['Step_1_Model_Output'] = calculation_df['Probability'].copy()
    
    
    calculation_df['Step_2_Sigmoid_Formula'] = "1 / (1 + e^(-x))"
    
   
    calculation_df['Step_3_Threshold'] = 0.5
    calculation_df['Step_4_Comparison'] = calculation_df['Probability'].apply(
        lambda x: f"{x:.4f} >= 0.5" if x >= 0.5 else f"{x:.4f} < 0.5"
    )
    
    
    calculation_df['Step_5_Decision'] = calculation_df['Prediction']
    
   
    calculation_df['Confidence_Score'] = calculation_df['Probability'].apply(
        lambda x: abs(x - 0.5) * 2  
    )
    
    calculation_df['Confidence_Level'] = calculation_df['Confidence_Score'].apply(
        lambda x: 'Very High (>80%)' if x > 0.8 else 
                  'High (60-80%)' if x > 0.6 else
                  'Medium (40-60%)' if x > 0.4 else
                  'Low (<40%)'
    )
    
    
    calculation_df['Math_Explanation'] = calculation_df.apply(
        lambda row: f"Model Output: {row['Probability']:.4f} → " +
                   f"Compare with 0.5 → {row['Prediction']} " +
                   f"(Confidence: {row['Confidence_Score']*100:.1f}%)",
        axis=1
    )
    
    
    final_df = calculation_df[[
        'Frame',
        'Step_1_Model_Output',
        'Step_3_Threshold', 
        'Step_4_Comparison',
        'Step_5_Decision',
        'Confidence_Score',
        'Confidence_Level',
        'Math_Explanation'
    ]]
    
    # Save to CSV
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/probability_calculation_breakdown.csv"
    final_df.to_csv(output_path, index=False)
    
    print("="*80)
    print(" PROBABILITY CALCULATION BREAKDOWN")
    print("="*80)
    print("\n HOW IT WORKS:\n")
    print("Step 1: EfficientNetB0 Model Processing")
    print("   - Input: 224x224 RGB retinal image")
    print("   - Process: Deep neural network feature extraction")
    print("   - Output: Raw prediction score\n")
    
    print("Step 2: Sigmoid Activation")
    print("   - Formula: P(disease) = 1 / (1 + e^(-x))")
    print("   - Converts raw score to probability (0 to 1)")
    print("   - Output: Disease probability\n")
    
    print("Step 3: Threshold Comparison")
    print("   - Threshold: 0.5")
    print("   - If P(disease) >= 0.5 → Diseased")
    print("   - If P(disease) < 0.5 → Normal\n")
    
    print("Step 4: Confidence Calculation")
    print("   - Confidence = |Probability - 0.5| × 2")
    print("   - Measures how far from decision boundary")
    print("   - Range: 0 (uncertain) to 1 (very confident)\n")
    
    print("="*80)
    print("📁 EXAMPLE CALCULATIONS:")
    print("="*80)
    
    # Show 5 examples
    for idx in range(min(5, len(df))):
        row = final_df.iloc[idx]
        prob = calculation_df.iloc[idx]['Probability']
        pred = calculation_df.iloc[idx]['Prediction']
        conf = calculation_df.iloc[idx]['Confidence_Score']
        
        print(f"\n📸 {row['Frame']}:")
        print(f"   Model Output: {prob:.4f}")
        print(f"   Comparison: {prob:.4f} {'≥' if prob >= 0.5 else '<'} 0.5")
        print(f"   Decision: {pred}")
        print(f"   Confidence: {conf*100:.1f}% ({calculation_df.iloc[idx]['Confidence_Level']})")
    
    print("\n" + "="*80)
    print(f"✅ Detailed breakdown saved to: {output_path}")
    print("="*80)
    
    # Create mathematical formula document
    create_formula_explanation()
    
    return final_df


def create_formula_explanation():
    
    
    output_path = "outputs/probability_formula_explanation.txt"
    with open(output_path, 'w') as f:
        f.write(formula_text)
    
    print(f"✅ Formula explanation saved to: {output_path}")


if __name__ == "__main__":
    explain_probability_calculation("data/predictions.csv")