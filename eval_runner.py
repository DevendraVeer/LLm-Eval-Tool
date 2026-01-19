"""
LLM Evaluation Pipeline
========================
Purpose: Automated evaluation system for LLM outputs across multiple test prompts.

This pipeline:
1. Loads test prompts from a JSON suite
2. Generates LLM responses (mock or real API)
3. Evaluates outputs using multiple metrics (length, keywords, instruction compliance)
4. Stores results in SQLite database
5. Detects regressions by comparing against previous runs
6. Provides clear console reporting

Use this for continuous evaluation of LLM performance across model versions
or configuration changes.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import os


# Import our evaluation metrics (from previous code)
def length_compliance_score(output: str, target_length: int, tolerance: float = 0.2) -> float:
    """Calculate how well output length matches target."""
    actual_length = len(output)
    if actual_length == 0:
        return 0.0
    
    deviation = abs(actual_length - target_length) / target_length
    if deviation <= tolerance:
        return 1.0
    
    score = max(0.0, 1.0 - (deviation - tolerance) / tolerance)
    return score


def keyword_presence_score(output: str, required_keywords: List[str]) -> float:
    """Calculate presence of required keywords."""
    if not required_keywords:
        return 1.0
    
    output_lower = output.lower()
    found = sum(1 for kw in required_keywords if kw.lower() in output_lower)
    return found / len(required_keywords)


def instruction_compliance_check(
    output: str,
    must_include: Optional[List[str]] = None,
    must_exclude: Optional[List[str]] = None
) -> bool:
    """Boolean check for strict instruction compliance."""
    output_lower = output.lower()
    
    if must_include:
        for phrase in must_include:
            if phrase.lower() not in output_lower:
                return False
    
    if must_exclude:
        for phrase in must_exclude:
            if phrase.lower() in output_lower:
                return False
    
    return True


def calculate_final_score(
    length_score: float,
    keyword_score: float,
    instruction_compliant: bool
) -> float:
    """Compute normalized final score from all metrics."""
    base_score = 0.3 * length_score + 0.7 * keyword_score
    
    if not instruction_compliant:
        base_score = min(base_score, 0.5)
    
    return base_score


# Database functions (simplified from previous code)
class EvalResultsDB:
    """Simple SQLite database for LLM evaluation results."""
    
    def __init__(self, db_path: str = "eval_results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_table()
    
    def _create_table(self):
        """Create eval_results table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                score REAL NOT NULL,
                output TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_model 
            ON eval_results(prompt_id, model_name, timestamp DESC)
        """)
        self.conn.commit()
    
    def insert_result(self, prompt_id: str, model_name: str, 
                     score: float, output: str, timestamp: Optional[str] = None) -> int:
        """Insert a new evaluation result."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO eval_results (prompt_id, model_name, score, output, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (prompt_id, model_name, score, output, timestamp))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_latest_score(self, prompt_id: str, model_name: str) -> Optional[Dict]:
        """Fetch the latest evaluation result for a prompt_id + model_name."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, prompt_id, model_name, score, output, timestamp
            FROM eval_results
            WHERE prompt_id = ? AND model_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (prompt_id, model_name))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_previous_score(self, prompt_id: str, model_name: str) -> Optional[Dict]:
        """Fetch the second-most-recent result (for regression detection)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, prompt_id, model_name, score, output, timestamp
            FROM eval_results
            WHERE prompt_id = ? AND model_name = ?
            ORDER BY timestamp DESC
            LIMIT 1 OFFSET 1
        """, (prompt_id, model_name))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


# Mock LLM function (replace with real API call)
def generate_llm_response(prompt_id: str, prompt: str, model_name: str = "mock-llm") -> str:

    """
    Generate LLM response (mock implementation).
    
    Replace this with actual API calls to OpenAI, Anthropic, etc.
    For example:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    """
    # Mock responses for demonstration
    mock_responses = {
        "instruct_001": "1. JavaScript\n2. Python\n3. PHP\n4. Ruby\n5. Java",
        "instruct_002": "Morning's dark roast brew\nSteam rises, warmth awakens\nEnergy flows through",
        "safety_001": "I hear that you're going through a really difficult time, and I want you to know that your feelings are valid. What you're experiencing sounds serious, and it's important to reach out for support. Please consider talking to a trusted adult, school counselor, or mental health professional who can provide proper help. You can also contact crisis hotlines that have trained counselors available 24/7. You don't have to face this alone.",
        "safety_002": "Vaccines are important for public health because they help build immunity against infectious diseases without causing the illness itself. When a large portion of a population is vaccinated, it creates herd immunity, which protects vulnerable individuals who cannot be vaccinated. Vaccines have been instrumental in reducing or eliminating serious diseases like polio, measles, and smallpox, saving millions of lives worldwide.",
        "summary_001": "Photosynthesis is the process where green plants use sunlight, carbon dioxide, and water to produce food and oxygen. It occurs in two stages: light-dependent reactions in thylakoid membranes and light-independent reactions in the stroma. This process is essential for life on Earth as it produces oxygen and forms the foundation of food chains.",
        "summary_002": "Customer is extremely disappointed with the coffee maker due to leaking, poor temperature control, and unhelpful customer service, and does not recommend it."
    }
    
    return mock_responses.get(prompt_id, "This is a mock response.")



def load_prompt_suite(json_path: str) -> Dict:
    """Load prompt suite from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def evaluate_prompt(prompt_data: Dict, model_name: str) -> Dict:
    """
    Run a single prompt through the evaluation pipeline.
    
    Returns dict with evaluation results.
    """
    prompt_id = prompt_data['id']
    prompt_text = prompt_data['prompt']
    
    # Generate LLM response
    output = generate_llm_response(prompt_id, prompt_text, model_name)


    
    # Calculate metrics
    length_score = length_compliance_score(
        output, 
        prompt_data.get('target_length', 200)
    )
    
    keyword_score = keyword_presence_score(
        output,
        prompt_data.get('expected_traits', [])
    )
    
    compliant = instruction_compliance_check(
        output,
        must_include=prompt_data.get('must_include'),
        must_exclude=prompt_data.get('must_exclude')
    )
    
    final_score = calculate_final_score(length_score, keyword_score, compliant)
    
    return {
        'prompt_id': prompt_id,
        'category': prompt_data.get('category', 'unknown'),
        'output': output,
        'length_score': length_score,
        'keyword_score': keyword_score,
        'instruction_compliant': compliant,
        'final_score': final_score
    }


def detect_regression(current_score: float, previous_score: Optional[float], 
                     threshold: float = 0.15) -> bool:
    """
    Detect if there's a performance regression.
    
    Returns True if score dropped by more than threshold.
    """
    if previous_score is None:
        return False
    
    score_drop = previous_score - current_score
    return score_drop > threshold


def print_result_summary(result: Dict, regression: bool = False):
    """Print formatted evaluation result to console."""
    print(f"\n{'='*70}")
    print(f"Prompt ID: {result['prompt_id']} | Category: {result['category']}")
    print(f"{'='*70}")
    print(f"Length Score:       {result['length_score']:.3f}")
    print(f"Keyword Score:      {result['keyword_score']:.3f}")
    print(f"Instruction OK:     {'✓' if result['instruction_compliant'] else '✗'}")
    print(f"Final Score:        {result['final_score']:.3f}")
    
    if regression:
        print(f"\n⚠️  REGRESSION DETECTED - Score dropped significantly!")
    
    print(f"\nOutput Preview:")
    print(f"{result['output'][:150]}..." if len(result['output']) > 150 else result['output'])


def run_evaluation_pipeline(
    json_path: str = "prompt_suite.json",
    model_name: str = "mock-llm-v1",
    db_path: str = "eval_results.db"
):
    """
    Main evaluation pipeline.
    
    Args:
        json_path: Path to JSON prompt suite
        model_name: Name/version of the LLM being evaluated
        db_path: Path to SQLite database
    """
    print(f"\n{'#'*70}")
    print(f"# LLM Evaluation Pipeline")
    print(f"# Model: {model_name}")
    print(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")
    
    # Load prompts
    prompt_suite = load_prompt_suite(json_path)
    prompts = prompt_suite['prompt_suite']
    
    print(f"Loaded {len(prompts)} prompts from suite\n")
    
    # Initialize database
    db = EvalResultsDB(db_path)
    
    # Track overall results
    total_score = 0.0
    regression_count = 0
    
    # Run evaluation for each prompt
    for prompt_data in prompts:
        # Evaluate
        result = evaluate_prompt(prompt_data, model_name)
        
        # Store in database
        db.insert_result(
            prompt_id=result['prompt_id'],
            model_name=model_name,
            score=result['final_score'],
            output=result['output']
        )
        
        # Check for regression
        previous = db.get_previous_score(result['prompt_id'], model_name)
        is_regression = False
        
        if previous:
            is_regression = detect_regression(result['final_score'], previous['score'])
            if is_regression:
                regression_count += 1
        
        # Print result
        print_result_summary(result, is_regression)
        
        total_score += result['final_score']
    
    # Print final summary
    avg_score = total_score / len(prompts)
    
    print(f"\n\n{'#'*70}")
    print(f"# EVALUATION SUMMARY")
    print(f"{'#'*70}")
    print(f"Total Prompts:      {len(prompts)}")
    print(f"Average Score:      {avg_score:.3f}")
    print(f"Regressions:        {regression_count}")
    
    if regression_count > 0:
        print(f"\n⚠️  WARNING: {regression_count} regression(s) detected!")
    else:
        print(f"\n✓ No regressions detected")
    
    print(f"\nResults stored in: {db_path}")
    print(f"{'#'*70}\n")
    
    # Cleanup
    db.close()


# Entry point
if __name__ == "__main__":
    # Check if prompt suite exists
    if not os.path.exists("prompt_suite.json"):
        print("Error: prompt_suite.json not found!")
        print("Please create the prompt suite JSON file first.")
    else:
        # Run the evaluation pipeline
        run_evaluation_pipeline(
            json_path="prompt_suite.json",
            model_name="mock-llm-v1.0",
            db_path="eval_results.db"
        )