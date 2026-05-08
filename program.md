# Token Merging Autoresearch Program

This program directs an AI agent to autonomously find the best token merging
configuration for ViT and VLM models.

## Goal

Find the (strategy, ratio) pair that maximizes the **quality-speed score**:

    score = cosine_sim * speedup

Higher is better. A config that perfectly preserves output quality (cos_sim=1.0)
at 2x speedup scores 2.0. A config that degrades quality to 0.8 at 3x speedup
scores 2.4 -- but only if the quality loss is acceptable (cosine_sim >= 0.95).

## Constraints

- `cosine_sim >= 0.95` -- reject configs that degrade output quality too much
- `pred_match >= 0.90` -- predictions must mostly agree with the unmerged model
- Only modify strategy and ratio; do not modify model weights or architecture

## Files

- `benchmark_tome.py` -- the evaluation script. Do NOT modify.
- `results.tsv` -- experiment log (tab-separated). You append to this.
- `analysis_tome.ipynb` -- visualization notebook. Do NOT modify.

## Experiment Loop

1. Pick a (strategy, ratio) config to test. Start with the grid:
   - Strategies: bipartite, kmeans, average_pool
   - Ratios: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
   After the grid, try fine-grained ratios around the best performers.

2. Run the benchmark:
   ```
   python benchmark_tome.py --strategy <s> --ratio <r> --model vit --runs 20 > run.log 2>&1
   ```

3. Parse results:
   ```
   grep "^cosine_sim:\|^speedup:\|^pred_match:\|^kl_div:\|^merged_ms:" run.log
   ```

4. Compute the score: `score = cosine_sim * speedup`

5. Log to results.tsv:
   ```
   strategy	ratio	cosine_sim	speedup	pred_match	kl_div	score	status	description
   ```
   - status: `keep` if score > current best AND constraints met, else `discard`

6. Repeat for VLM: `--model vlm`

## NEVER STOP

Run all grid combinations, then refine. Do not ask the human for permission.
