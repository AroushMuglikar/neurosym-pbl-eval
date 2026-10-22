import numpy as np
from scipy.optimize import minimize
import yaml
import logging

class SymbolicAlignmentEngine:
    """
    Executes the convex quadratic projection solver (Section 7) to align unconstrained 
    LLM score proposals with observable multimodal telemetry constraints.
    """
    def __init__(self, config_path="configs/eval_thresholds.yaml", num_criteria=4):
        self.num_criteria = num_criteria
        # Initialize identity inverse covariance matrix; to be updated via calibration
        self.cov_matrix_inv = np.eye(self.num_criteria)
        
        try:
            with open(config_path, 'r') as file:
                self.thresholds = yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Config {config_path} not found. Operating with symbolic defaults.")
            self.thresholds = self._get_fallback_thresholds()

    def _get_fallback_thresholds(self):
        return {
            'sophias_individual': {
                'construct_a_gaze': {'h_gaze_min_entropy': 0.28, 'note_fixation_max_ratio': 0.65, 'hard_ceiling_score': 2.0},
                'construct_b_cognitive_load': {'rmssd_z_threshold': -1.50, 'f_pause_min_ratio': 0.12, 'conjunctive_penalty_weight': 0.75}
            },
            'pehri_collaborative': {
                'construct_c_equity': {'g_speech_max_gini': 0.72, 'hard_ceiling_score': 2.5},
                'construct_d_joint_attention': {'r_add_del_min': 0.40, 'crq_gaze_min': 0.15, 'hard_ceiling_score': 3.0}
            }
        }

    def _qp_objective(self, y, y_tilde, penalties):
        """
        Objective: 0.5 * (y - y_tilde)^T * Sigma^{-1} * (y - y_tilde) + c^T * y
        """
        diff = y - y_tilde
        base_loss = 0.5 * np.dot(diff.T, np.dot(self.cov_matrix_inv, diff))
        penalty_loss = np.sum(penalties * y)
        return base_loss + penalty_loss

    def project_scores(self, llm_scores, extracted_features, quality_masks):
        """
        llm_scores: Array of raw scores proposed by the LLM [1.0 to 5.0]
        extracted_features: Dict of mapped features (e.g., 'h_gaze', 'g_speech')
        quality_masks: Dict of sensor validity flags {modality: boolean} (Tier 3 dropout abstention)
        """
        bounds = [[1.0, 5.0] for _ in range(self.num_criteria)]
        penalties = np.zeros(self.num_criteria)
        active_constraints = []
        
        # ---------------------------------------------------------
        # Tier 1: Hard Score Ceilings (Objective Allocations)
        # ---------------------------------------------------------
        cfg_gaze = self.thresholds['sophias_individual']['construct_a_gaze']
        if quality_masks.get('gaze', True):
            if extracted_features.get('h_gaze', 1.0) < cfg_gaze['h_gaze_min_entropy'] and \
               extracted_features.get('r_notes', 0.0) > cfg_gaze['note_fixation_max_ratio']:
                bounds[0][1] = cfg_gaze['hard_ceiling_score']
                active_constraints.append('Tier 1: Gaze Fixation Ceiling')
                
        cfg_gini = self.thresholds['pehri_collaborative']['construct_c_equity']
        if quality_masks.get('audio', True):
            if extracted_features.get('g_speech', 0.0) > cfg_gini['g_speech_max_gini']:
                bounds[2][1] = cfg_gini['hard_ceiling_score']
                active_constraints.append('Tier 1: Speech Monopoly Ceiling')

        # ---------------------------------------------------------
        # Tier 2: Conjunctive Soft Penalties (Ambiguous Signals)
        # ---------------------------------------------------------
        cfg_load = self.thresholds['sophias_individual']['construct_b_cognitive_load']
        if quality_masks.get('ppg', True) and quality_masks.get('audio', True):
            if extracted_features.get('rmssd_z', 0.0) < cfg_load['rmssd_z_threshold'] and \
               extracted_features.get('f_pause', 0.0) > cfg_load['f_pause_min_ratio']:
                penalties[1] = cfg_load['conjunctive_penalty_weight']
                active_constraints.append('Tier 2: Conjunctive Overload Penalty')

        # Execute SLSQP Solver
        result = minimize(
            self._qp_objective,
            x0=llm_scores,
            args=(llm_scores, penalties),
            bounds=bounds,
            method='SLSQP',
            options={'ftol': 1e-6, 'disp': False}
        )
        
        # Calculate Mahalanobis projection distance
        projected_scores = np.clip(result.x, 1.0, 5.0)
        proj_distance = np.dot((projected_scores - llm_scores).T, 
                               np.dot(self.cov_matrix_inv, (projected_scores - llm_scores)))

        audit_record = {
            "success": result.success,
            "original_scores": llm_scores.tolist(),
            "projected_scores": projected_scores.tolist(),
            "active_constraints": active_constraints,
            "projection_distance_sq": float(proj_distance)
        }
        
        return projected_scores, audit_record
