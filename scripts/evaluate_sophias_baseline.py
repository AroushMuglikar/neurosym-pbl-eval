import sys
from pathlib import Path

# Add src to path for relative imports
sys.path.append(str(Path(__file__).parent.parent))

from src.memory.episodic_buffer import EpisodicMemoryBuffer
from src.symbolic.logic_engine import AlignmentEngine
from data_loaders.sophias_ingestion import SOPHIASDataLoader

def run_baseline_evaluation(student_id: str = "SJTU_TEST_001"):
    print(f"--- Initializing Neuro-Symbolic Pipeline for {student_id} ---\n")
    
    # 1. Initialize Modules
    loader = SOPHIASDataLoader(data_dir="../data/raw_sophias/")
    # Using a 10-window buffer (100 seconds of memory) with high decay
    memory_buffer = EpisodicMemoryBuffer(temporal_window=10, decay_rate=0.85)
    evaluator = AlignmentEngine(config_path="../../configs/eval_thresholds.yaml")
    
    # 2. Ingest Data
    print("[INFO] Ingesting multimodal temporal streams...")
    session_data = loader.load_student_session(student_id)
    
    # 3. Simulate streaming the presentation data chronologically
    print("[INFO] Populating Episodic Memory Buffer...")
    for idx, row in session_data.iterrows():
        # Feed the neural feature state into memory
        feature_state = {
            'gaze_score': row['gaze_score'],
            'fluency_score': row['fluency_score']
        }
        memory_buffer.update(feature_state)
        
        # Every 20 windows (approx 3 minutes), evaluate the state
        if idx > 0 and idx % 20 == 0:
            aggregated_context = memory_buffer.get_aggregated_context()
            print(f"\n  [Time: {idx*10}s] Aggregated Memory State: "
                  f"Gaze={aggregated_context['gaze_score']:.2f}, "
                  f"Fluency={aggregated_context['fluency_score']:.2f}")
            
            # 4. Symbolic Alignment Check
            status = evaluator.assess_communication(aggregated_context)
            print(f"  [Symbolic Engine]: {status}")

if __name__ == "__main__":
    run_baseline_evaluation()
