import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.memory.episodic_buffer import EpisodicMemoryBuffer

def grid_search_decay_rates():
    """
    Brute-force optimization to find the optimal temporal decay factor (lambda).
    We want to retain enough history for context, but decay fast enough to catch 
    sudden drops in student engagement.
    """
    print("--- Initiating Hyperparameter Search for Episodic Decay ---")
    candidate_rates = np.arange(0.5, 0.99, 0.05)
    best_rate = None
    min_variance = float('inf')
    
    # Simulating a highly volatile 2-minute presentation segment
    volatile_stream = [{'gaze_score': np.random.uniform(0.2, 0.9)} for _ in range(120)]
    
    for rate in candidate_rates:
        buffer = EpisodicMemoryBuffer(temporal_window=10, decay_rate=rate)
        aggregated_scores = []
        
        for state in volatile_stream:
            buffer.update(state)
            agg = buffer.get_aggregated_context()
            if agg:
                aggregated_scores.append(agg['gaze_score'])
                
        # Calculate smoothing stability
        variance = np.var(aggregated_scores)
        print(f"Testing Lambda: {rate:.2f} -> Memory Smoothing Variance: {variance:.4f}")
        
        if variance < min_variance:
            min_variance = variance
            best_rate = rate
            
    print(f"\n[RESULT] Optimal decay rate bounded at ~{best_rate:.2f} for high-volatility streams.")
    print("Update configs/model_config.yaml with this value before running full alignment.")

if __name__ == "__main__":
    grid_search_decay_rates()
