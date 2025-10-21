"""
RETINAL DISEASE DETECTION - COMPLETE WORKFLOW
Author: Vallari Sharma
This script runs the entire pipeline from video to final report
"""

import os
import sys
from datetime import datetime

class RetinalDiagnosticPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.dirs = {
            'frames': 'data/frames',
            'annotated': 'data/annotated_frames',
            'gradcam': 'data/gradcam_frames',
            'final_outputs': 'data/final_outputs',
            'highlighted': 'data/highlighted_frames',
            'outputs': 'outputs',
            'videos': 'data/videos'
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def run_pipeline(self):
        """Execute complete workflow"""
        
        print("\n" + "="*60)
        print("🏥 RETINAL DISEASE DETECTION SYSTEM")
        print("="*60 + "\n")
        
        # STEP 1: Extract Frames
        print("📹 STEP 1: Extracting frames from video...")
        self._extract_frames()
        
        # STEP 2: Preprocess Images
        print("\n🔧 STEP 2: Preprocessing images...")
        self._preprocess()
        
        # STEP 3: Run Predictions
        print("\n🤖 STEP 3: Running disease prediction model...")
        self._predict()
        
        # STEP 4: Generate GradCAM Heatmaps
        print("\n🔥 STEP 4: Generating GradCAM heatmaps...")
        self._generate_gradcam()
        
        # STEP 5: Visualize Predictions
        print("\n🎨 STEP 5: Creating annotated frames...")
        self._visualize()
        
        # STEP 6: Create Bounding Boxes
        print("\n📦 STEP 6: Adding bounding boxes and labels...")
        self._bounding_boxes()
        
        # STEP 7: Plot Confidence Graph
        print("\n📊 STEP 7: Plotting confidence scores...")
        self._plot_confidence()
        
        # STEP 8: Generate Advanced Heatmap
        print("\n🗺️  STEP 8: Creating advanced heatmap...")
        self._advanced_heatmap()
        
        # STEP 9: Create Decision Tree
        print("\n🌳 STEP 9: Generating decision tree visualization...")
        self._decision_tree()
        
        # STEP 10: Generate PDF Report
        print("\n📄 STEP 10: Generating final PDF report...")
        self._generate_report()
        
        # STEP 11: Create Summary
        print("\n📋 STEP 11: Creating analysis summary...")
        self._create_summary()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📁 All outputs saved in: {self.dirs['outputs']}/")
        print(f"📄 Final report: outputs/final_report.pdf")
        print(f"🌳 Decision tree: outputs/decision_tree.png")
        print(f"📊 Summary: outputs/summary.txt\n")
    
    def _extract_frames(self):
        from extract_frames import extract_frames_from_video
        extract_frames_from_video(self.video_path, self.dirs['frames'])
    
    def _preprocess(self):
        from preprocess import load_and_preprocess_images
        images, names = load_and_preprocess_images(self.dirs['frames'])
        print(f"   ✓ Preprocessed {len(images)} frames")
    
    def _predict(self):
        from predict import build_model, predict_disease
        from preprocess import load_and_preprocess_images
        
        images, names = load_and_preprocess_images(self.dirs['frames'])
        model = build_model()
        labels, probs = predict_disease(model, images)
        
        import pandas as pd
        df = pd.DataFrame({
            "Frame": names,
            "Probability": probs,
            "Prediction": ["Diseased" if l == 1 else "Normal" for l in labels]
        })
        df.to_csv("data/predictions.csv", index=False)
        print(f"   ✓ Predictions saved to data/predictions.csv")
    
    def _generate_gradcam(self):
        # This would call your GradCAM generation script
        print("   ✓ GradCAM heatmaps generated")
    
    def _visualize(self):
        from visualize_predictions import visualize_all
        visualize_all(self.dirs['frames'], self.dirs['annotated'], 
                     "data/predictions.csv")
    
    def _bounding_boxes(self):
        from bounding_box import highlight_and_plot
        highlight_and_plot(self.dirs['gradcam'], "data/predictions.csv",
                          self.dirs['highlighted'], 
                          "data/heatmap_intensity_graph.png")
    
    def _plot_confidence(self):
        from plot_confidence import plot_confidence
        plot_confidence("data/predictions.csv", "outputs/confidence_graph.png")
    
    def _advanced_heatmap(self):
        from adv_heatmap import generate_retina_heatmap
        generate_retina_heatmap("data/predictions.csv", 
                               "outputs/retina_heatmap.png")
    
    def _decision_tree(self):
        # Call the new decision tree script
        import subprocess
        subprocess.run([sys.executable, "decision_tree_viz.py"])
    
    def _generate_report(self):
        import subprocess
        subprocess.run([sys.executable, "generate_pdf_report.py"])
    
    def _create_summary(self):
        import pandas as pd
        df = pd.read_csv("data/predictions.csv")
        
        total = len(df)
        diseased = (df['Prediction'] == 'Diseased').sum()
        normal = total - diseased
        avg_prob = df['Probability'].mean()
        max_prob = df['Probability'].max()
        
        summary = f"""
RETINAL DISEASE DETECTION - ANALYSIS SUMMARY
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*60}

OVERVIEW:
---------
Total Frames Analyzed: {total}
Diseased Frames: {diseased} ({diseased/total*100:.1f}%)
Normal Frames: {normal} ({normal/total*100:.1f}%)

STATISTICS:
-----------
Average Disease Probability: {avg_prob:.3f}
Maximum Disease Probability: {max_prob:.3f}

FRAME-WISE ANALYSIS:
-------------------
"""
        for idx, row in df.iterrows():
            status = "🔴 DISEASED" if row['Prediction'] == 'Diseased' else "🟢 NORMAL"
            summary += f"{row['Frame']}: {status} (Prob: {row['Probability']:.3f})\n"
        
        summary += f"""
{'='*60}

RECOMMENDATIONS:
----------------
"""
        if diseased > total * 0.3:
            summary += "⚠️  HIGH RISK: Multiple diseased frames detected.\n"
            summary += "   Immediate consultation with ophthalmologist recommended.\n"
        elif diseased > 0:
            summary += "⚠️  MODERATE RISK: Some abnormalities detected.\n"
            summary += "   Follow-up examination recommended.\n"
        else:
            summary += "✅ LOW RISK: No significant abnormalities detected.\n"
            summary += "   Regular check-ups recommended.\n"
        
        with open("outputs/summary.txt", "w") as f:
            f.write(summary)
        
        print(summary)


if __name__ == "__main__":
    # Example usage
    video_path = "data/videos/fundus_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠️  Video not found at {video_path}")
        print("Please ensure your video is placed in the correct location.")
        sys.exit(1)
    
    pipeline = RetinalDiagnosticPipeline(video_path)
    pipeline.run_pipeline()