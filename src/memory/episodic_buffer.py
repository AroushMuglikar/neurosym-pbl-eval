import numpy as np
from typing import List, Dict

class EpisodicMemoryBuffer:
    """
    Temporal memory module to track engineering behavior across windows.
    Adapted from the MambaGBA episodic concept. We can't assess PBL in a vacuum.
    """
    def __init__(self, temporal_window: int = 10, decay_rate: float = 0.9):
        self.window = temporal_window
        self.decay_rate = decay_rate
        self.state_history: List[Dict[str, float]] = []
        
    def update(self, feature_state: Dict[str, float]):
        """
        Pushes a new 10-second multimodal window into the buffer.
        """
        if len(self.state_history) >= self.window:
            self.state_history.pop(0) # drop oldest
        self.state_history.append(feature_state)
        
    def get_aggregated_context(self) -> dict:
        """
        Calculates weighted average of recent states.
        # FIXME: linear decay is probably too naive here. Might need an LSTM 
        # or attention mechanism later if the variance in PE-HRI logs is too high.
        """
        if not self.state_history:
            return {}
            
        keys = self.state_history[0].keys()
        aggregated = {}
        
        # Apply simple decay so recent actions matter more than older ones
        weights = [self.decay_rate ** i for i in range(len(self.state_history))][::-1]
        weight_sum = sum(weights)
        
        for k in keys:
            vals = [state[k] for state in self.state_history]
            aggregated[k] = np.dot(vals, weights) / weight_sum
            
        return aggregated
