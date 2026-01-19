"""
Simple evaluation metrics for LLM outputs.
Assumptions:
- Length compliance: measures how close output is to target length (within ±20% is ideal)
- Keyword presence: checks for required keywords/traits (case-insensitive)
- Instruction compliance: boolean check for must-have/must-not-have conditions
- Final score: weighted average of all metrics
"""

import re
from typing import List, Dict, Optional


def length_compliance_score(
    output: str,
    target_length: int,
    tolerance: float = 0.2
) -> float:
    """
    Calculate how well output length matches target.
    
    Args:
        output: LLM output text
        target_length: desired length (in characters or words)
        tolerance: acceptable deviation (default 20%)
    
    Returns:
        Score from 0-1, where 1 is perfect compliance
    
    Assumption: Uses character count by default. For word count,
    pass len(output.split()) as actual length.
    """
    actual_length = len(output)
    
    if actual_length == 0:
        return 0.0
    
    # Calculate deviation from target
    deviation = abs(actual_length - target_length) / target_length
    
    # Within tolerance gets full score
    if deviation <= tolerance:
        return 1.0
    
    # Linear decay beyond tolerance, reaches 0 at 2x tolerance
    score = max(0.0, 1.0 - (deviation - tolerance) / tolerance)
    return score


def keyword_presence_score(
    output: str,
    required_keywords: List[str],
    optional_keywords: Optional[List[str]] = None
) -> float:
    """
    Calculate presence of required and optional keywords.
    
    Args:
        output: LLM output text
        required_keywords: must be present (weighted 70%)
        optional_keywords: nice to have (weighted 30%)
    
    Returns:
        Score from 0-1 based on keyword presence
    
    Assumption: Case-insensitive matching, partial word matches count.
    """
    output_lower = output.lower()
    
    # Check required keywords
    if required_keywords:
        required_found = sum(
            1 for kw in required_keywords 
            if kw.lower() in output_lower
        )
        required_score = required_found / len(required_keywords)
    else:
        required_score = 1.0
    
    # Check optional keywords
    if optional_keywords:
        optional_found = sum(
            1 for kw in optional_keywords 
            if kw.lower() in output_lower
        )
        optional_score = optional_found / len(optional_keywords)
    else:
        optional_score = 1.0
    
    # Weighted combination: 70% required, 30% optional
    final_score = 0.7 * required_score + 0.3 * optional_score
    return final_score


def instruction_compliance_check(
    output: str,
    must_include: Optional[List[str]] = None,
    must_exclude: Optional[List[str]] = None,
    format_pattern: Optional[str] = None
) -> bool:
    """
    Boolean check for strict instruction compliance.
    
    Args:
        output: LLM output text
        must_include: phrases that MUST be present
        must_exclude: phrases that MUST NOT be present
        format_pattern: regex pattern output must match
    
    Returns:
        True if all conditions met, False otherwise
    
    Assumption: All conditions are AND-ed together (all must pass).
    """
    output_lower = output.lower()
    
    # Check must-include conditions
    if must_include:
        for phrase in must_include:
            if phrase.lower() not in output_lower:
                return False
    
    # Check must-exclude conditions
    if must_exclude:
        for phrase in must_exclude:
            if phrase.lower() in output_lower:
                return False
    
    # Check format pattern (e.g., JSON structure, specific format)
    if format_pattern:
        if not re.search(format_pattern, output, re.DOTALL):
            return False
    
    return True


def calculate_final_score(
    length_score: float,
    keyword_score: float,
    instruction_compliant: bool,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute normalized final score from all metrics.
    
    Args:
        length_score: score from length_compliance_score (0-1)
        keyword_score: score from keyword_presence_score (0-1)
        instruction_compliant: boolean from instruction_compliance_check
        weights: optional custom weights for each component
    
    Returns:
        Final normalized score from 0-1
    
    Assumption: If instruction compliance fails, max score is capped at 0.5
    to reflect critical failure. Default weights: 30% length, 70% keywords.
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'length': 0.3,
            'keywords': 0.7
        }
    
    # Calculate weighted score
    base_score = (
        weights['length'] * length_score +
        weights['keywords'] * keyword_score
    )
    
    # If instruction compliance failed, cap the score
    if not instruction_compliant:
        base_score = min(base_score, 0.5)
    
    return base_score


# Example usage
if __name__ == "__main__":
    # Sample LLM output
    sample_output = """
    Python is a high-level programming language known for its simplicity 
    and readability. It's widely used in web development, data science, 
    and machine learning applications.
    """
    
    # Evaluate the output
    length_score = length_compliance_score(
        sample_output,
        target_length=200,
        tolerance=0.2
    )
    
    keyword_score = keyword_presence_score(
        sample_output,
        required_keywords=['python', 'programming'],
        optional_keywords=['data science', 'machine learning', 'web']
    )
    
    compliant = instruction_compliance_check(
        sample_output,
        must_include=['python'],
        must_exclude=['javascript', 'java'],
    )
    
    final_score = calculate_final_score(
        length_score,
        keyword_score,
        compliant
    )
    
    # Print results
    print(f"Length Compliance: {length_score:.2f}")
    print(f"Keyword Presence: {keyword_score:.2f}")
    print(f"Instruction Compliant: {compliant}")
    print(f"Final Score: {final_score:.2f}")