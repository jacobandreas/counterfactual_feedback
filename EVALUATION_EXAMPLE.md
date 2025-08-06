# Example Evaluation Results

This shows an example of what gets saved in `results/evaluation_{model1}_vs_{model2}_{timestamp}.json` after running the evaluation script:

```json
{
  "metadata": {
    "model1": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "model2": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "judge_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "start_time": "2025-07-25T10:30:15.123456",
    "end_time": "2025-07-25T11:45:32.654321",
    "duration_seconds": 4517.5,
    "total_conversations": 1000
  },
  "summary": {
    "model1_wins": 423,
    "model2_wins": 487,
    "ties": 78,
    "errors": 12,
    "total_evaluations": 988,
    "success_rate": 0.988
  },
  "statistics": {
    "total_evaluations": 988,
    "model1_win_rate": 0.428,
    "model2_win_rate": 0.493,
    "tie_rate": 0.079,
    "model1_confidence_interval_95": [0.396, 0.461],
    "model2_confidence_interval_95": [0.460, 0.526],
    "non_tie_evaluations": 910,
    "statistical_test": {
      "test": "binomial_test",
      "null_hypothesis": "models are equally good",
      "p_value": 0.0234,
      "significant_at_0.05": true,
      "interpretation": "Model 2 significantly better than Model 1"
    }
  },
  "evaluations": [
    {
      "conversation_id": "66440c86dd6841eda272f2ea431e2a6a",
      "status": "success",
      "context": [
        {
          "role": "user",
          "content": "What's the capital of France?"
        }
      ],
      "model1_response": "The capital of France is Paris.",
      "model2_response": "Paris is the capital and largest city of France, known for its art, fashion, and culture.",
      "original_response": "Paris is the capital of France.",
      "judgment": "B",
      "winner": "model2",
      "order": "normal"
    },
    {
      "conversation_id": "another_conversation_id",
      "status": "success",
      "context": [
        {
          "role": "user", 
          "content": "How do I bake a cake?"
        }
      ],
      "model1_response": "Mix ingredients and bake in oven.",
      "model2_response": "To bake a cake: 1) Preheat oven to 350°F, 2) Mix dry ingredients, 3) Add wet ingredients, 4) Pour into greased pan, 5) Bake 25-30 minutes.",
      "original_response": "You need flour, eggs, sugar, and butter. Mix them and bake.",
      "judgment": "B", 
      "winner": "model2",
      "order": "reversed"
    }
  ]
}
```

## Key Fields Explained:

### Metadata
- **model1/model2**: The two models being compared
- **judge_model**: Model used to make judgments
- **timing**: Start/end times and total duration
- **total_conversations**: Number of conversations attempted

### Summary Statistics
- **wins/ties/errors**: Count of each outcome type
- **total_evaluations**: Successfully completed evaluations
- **success_rate**: Percentage of conversations successfully evaluated

### Statistical Analysis
- **win_rates**: Proportion of wins for each model
- **confidence_intervals**: 95% confidence intervals for win rates
- **statistical_test**: Significance test results
  - **p_value**: Probability that observed difference is due to chance
  - **significant_at_0.05**: Whether difference is statistically significant
  - **interpretation**: Human-readable explanation of results

### Individual Evaluations
For each conversation:
- **context**: The conversation context sent to both models
- **model responses**: Generated responses from both models
- **original_response**: Original response from the conversation
- **judgment**: Judge's decision ("A", "B", or "TIE")
- **winner**: Which model won ("model1", "model2", or "tie")
- **order**: Whether models were presented in normal or reversed order

## Statistical Interpretation

### Win Rates and Confidence Intervals
- **Win Rate**: Percentage of non-tie evaluations won by each model
- **Confidence Interval**: Range of plausible values for the true win rate
- Narrower intervals indicate more precise estimates (larger sample sizes)

### Statistical Significance
- **P-value < 0.05**: Strong evidence that one model is better than the other
- **P-value ≥ 0.05**: Insufficient evidence to conclude one model is better
- **Confidence intervals**: If intervals don't overlap, models likely perform differently

### Example Interpretations

**Scenario 1: Clear Winner**
```
Model A: 65% wins (CI: 62%-68%)
Model B: 35% wins (CI: 32%-38%) 
P-value: 0.001
→ Model A is significantly better
```

**Scenario 2: No Clear Difference**
```
Model A: 52% wins (CI: 47%-57%)
Model B: 48% wins (CI: 43%-53%)
P-value: 0.234
→ No significant difference between models
```

**Scenario 3: Close but Significant**
```
Model A: 55% wins (CI: 52%-58%)
Model B: 45% wins (CI: 42%-48%)
P-value: 0.032
→ Model A slightly but significantly better
```

## Usage for Research

This evaluation framework enables:

1. **Model Development**: Compare different training approaches
2. **Ablation Studies**: Test impact of specific changes
3. **Benchmark Comparisons**: Evaluate against baseline models
4. **Statistical Rigor**: Confidence intervals and significance testing
5. **Detailed Analysis**: Access to all individual evaluations for deeper investigation
