from Real_Time_Smart_Grocery_Recommendation_System.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("Starting Real-Time Smart Grocery Recommendation System Pipeline")
        print("="*70 + "\n")
        
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        
        print("\n" + "="*70)
        print("✓ All pipelines completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Pipeline Error: {str(e)}")
        import traceback
        traceback.print_exc()