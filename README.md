# NeuroSym-PBL-Eval

WIP implementation of a neuro-symbolic evaluator for multimodal Project-Based Learning (PBL) data. 

The core idea here is to see if we can constrain LLM hallucinations during engineering assessments by hooking up a symbolic logic engine to an episodic memory buffer (heavily adapting the memory module concepts from MambaGBA). Pure LLM scoring on engineering rubrics is too unstable; this repo forces the neural outputs through hard deterministic thresholds.

Currently mapping the architecture to handle ingestion from the SOPHIAS and PE-HRI datasets.

## Repo Structure
* `configs/` - YAML files mapping out human-expert rubric thresholds and encoder weights.
* `src/encoders/` - Neural extraction stubs. Waiting on local data pulls to finish building out the video/audio feature extractors.
* `src/memory/` - The episodic buffer. Tracks student behavior over temporal windows rather than evaluating stateless snapshots.
* `src/symbolic/` - The rule engine. If the neural layer says a student's code is great but the commit history complexity is 0, this layer overrides the LLM grade.

## Next Steps
- [x] Dataset audit and pipeline architecture mapped.
- [ ] Write the actual data ingestion scripts for SOPHIAS PPG/Eye-tracking logs.
- [ ] Tune the temporal decay factor in the episodic buffer.
- [ ] Run baseline alignment tests against expert human scoring.
