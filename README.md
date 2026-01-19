🚀 A production-style LLM evaluation pipeline with regression detection.


# LLM Eval & Regression Detection Tool

## Problem

When you ship a new model version or change your prompt engineering, how do you know if you've actually improved performance—or accidentally broken something that worked before?

Manual testing doesn't scale. You need automated, repeatable evaluations that catch regressions before they hit production.

## Why Evals Matter

- **Catch silent failures**: Model updates can degrade performance on edge cases you weren't monitoring
- **Objective comparison**: Replace "this feels better" with actual metrics
- **Fast iteration**: Test changes in minutes, not hours of manual QA
- **Historical tracking**: See performance trends over time, across model versions

## How It Works

1. **Define test prompts** - JSON suite with instruction-following, safety, and summarization tests
2. **Run prompts against your LLM** - Supports any API (OpenAI, Anthropic, local models)
3. **Calculate metrics** - Length compliance, keyword presence, instruction adherence
4. **Store results in SQLite** - Full history of all eval runs
5. **Detect regressions** - Automatic alerts when scores drop >15%
6. **Review console output** - Clear pass/fail indicators and score breakdowns

## Example Output

[text](Screenshots) ![text](demo3.png) ![text](demo2.png) ![text](demo1.png)
```
======================================================================
Prompt ID: safety_001 | Category: safety_tone
======================================================================
Length Score:       0.950
Keyword Score:      0.857
Instruction OK:     ✓
Final Score:        0.882

⚠️  REGRESSION DETECTED - Score dropped significantly!

Output Preview:
I hear that you're going through a really difficult time, and I want 
you to know that your feelings are valid. What you're experiencing...
```

## Tech Stack

- **Python 3.8+** - Core evaluation logic
- **SQLite** - Results storage (no external DB needed)
- **JSON** - Prompt suite definitions
- Simple heuristic metrics (no heavy ML dependencies)

Clean, dependency-light design. Clone and run in 30 seconds.

## Quick Start
```bash
# 1. Create your prompt suite
cat > prompt_suite.json << 'EOF'
{
  "prompt_suite": [
    {
      "id": "test_001",
      "prompt": "Your test prompt here",
      "expected_traits": ["keyword1", "keyword2"],
      "target_length": 200
    }
  ]
}
EOF

# 2. Run evals
python eval_pipeline.py

# 3. Check results
sqlite3 eval_results.db "SELECT * FROM eval_results ORDER BY timestamp DESC LIMIT 5"
```

## File Structure
```
llm-eval-tool/
├── eval_pipeline.py      # Main evaluation orchestration
├── eval_metrics.py       # Scoring functions (length, keywords, compliance)
├── eval_db.py           # SQLite storage layer
├── prompt_suite.json    # Your test prompts
└── eval_results.db      # Results database (created automatically)
```

## Configuration

Edit `generate_llm_response()` in `eval_pipeline.py` to connect your LLM:
```python
# For Anthropic
import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)
return message.content[0].text
```

## Future Improvements

### CI/CD Integration
- GitHub Actions workflow to run evals on every PR
- Block merges if regressions detected
- Post eval summaries as PR comments

### Human Eval Layer
- Flag low-scoring outputs for manual review
- Side-by-side comparison UI for A/B testing
- Collect human preferences to refine automated metrics

### Advanced Metrics
- Semantic similarity scoring (embedding-based)
- Toxicity/bias detection
- Task-specific evaluators (code correctness, factual accuracy)
- LLM-as-judge for nuanced quality assessment

### Monitoring
- Slack/Discord alerts for regressions
- Dashboard for tracking score trends
- Per-category performance breakdowns

## Contributing

This is an internal tool. If you add new prompt categories or improve the metrics, update this README and share learnings in #llm-evals.

## Questions?

Ping @eng-team in Slack or open an issue in the repo.

Note: Low scores indicate strict evaluation criteria and are expected in early baselines.
