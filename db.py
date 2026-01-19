"""
SQLite database for storing LLM evaluation results.
Simple interface using sqlite3 (no ORM).
"""

import sqlite3
from datetime import datetime
from typing import Optional, Dict, List


class EvalResultsDB:
    """Simple SQLite database for LLM evaluation results."""
    
    def __init__(self, db_path: str = "eval_results.db"):
        """
        Initialize database connection and create table if needed.
        
        Args:
            db_path: path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
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
        
        # Create index for faster lookups on prompt_id + model_name
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_model 
            ON eval_results(prompt_id, model_name, timestamp DESC)
        """)
        
        self.conn.commit()
    
    def insert_result(
        self,
        prompt_id: str,
        model_name: str,
        score: float,
        output: str,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Insert a new evaluation result.
        
        Args:
            prompt_id: unique identifier for the prompt
            model_name: name of the LLM model
            score: evaluation score (0-1)
            output: the LLM's output text
            timestamp: optional timestamp (ISO format), defaults to now
        
        Returns:
            The id of the inserted row
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO eval_results (prompt_id, model_name, score, output, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (prompt_id, model_name, score, output, timestamp))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_latest_score(
        self,
        prompt_id: str,
        model_name: str
    ) -> Optional[Dict]:
        """
        Fetch the latest evaluation result for a prompt_id + model_name.
        
        Args:
            prompt_id: unique identifier for the prompt
            model_name: name of the LLM model
        
        Returns:
            Dict with keys: id, prompt_id, model_name, score, output, timestamp
            Returns None if no results found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, prompt_id, model_name, score, output, timestamp
            FROM eval_results
            WHERE prompt_id = ? AND model_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (prompt_id, model_name))
        
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Convert Row object to dict
        return dict(row)
    
    def get_all_results(
        self,
        prompt_id: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch evaluation results with optional filtering.
        
        Args:
            prompt_id: filter by prompt_id (optional)
            model_name: filter by model_name (optional)
            limit: maximum number of results to return
        
        Returns:
            List of dicts, ordered by most recent first
        """
        cursor = self.conn.cursor()
        
        # Build query dynamically based on filters
        query = "SELECT id, prompt_id, model_name, score, output, timestamp FROM eval_results"
        params = []
        conditions = []
        
        if prompt_id:
            conditions.append("prompt_id = ?")
            params.append(prompt_id)
        
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_average_score(
        self,
        prompt_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Optional[float]:
        """
        Calculate average score with optional filtering.
        
        Args:
            prompt_id: filter by prompt_id (optional)
            model_name: filter by model_name (optional)
        
        Returns:
            Average score, or None if no results
        """
        cursor = self.conn.cursor()
        
        query = "SELECT AVG(score) as avg_score FROM eval_results"
        params = []
        conditions = []
        
        if prompt_id:
            conditions.append("prompt_id = ?")
            params.append(prompt_id)
        
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        
        return result['avg_score'] if result['avg_score'] is not None else None
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connection."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Use context manager for automatic cleanup
    with EvalResultsDB("eval_results.db") as db:
        
        # Insert some sample results
        db.insert_result(
            prompt_id="prompt_001",
            model_name="gpt-4",
            score=0.85,
            output="This is a great response from GPT-4."
        )
        
        db.insert_result(
            prompt_id="prompt_001",
            model_name="claude-3",
            score=0.92,
            output="This is an excellent response from Claude."
        )
        
        db.insert_result(
            prompt_id="prompt_001",
            model_name="gpt-4",
            score=0.88,
            output="This is another good response from GPT-4."
        )
        
        # Get latest score for specific prompt + model
        latest = db.get_latest_score("prompt_001", "gpt-4")
        if latest:
            print(f"Latest GPT-4 score: {latest['score']}")
            print(f"Timestamp: {latest['timestamp']}")
            print(f"Output: {latest['output'][:50]}...")
        
        print("\n" + "="*50 + "\n")
        
        # Get all results for a prompt
        all_results = db.get_all_results(prompt_id="prompt_001")
        print(f"All results for prompt_001: {len(all_results)} entries")
        for result in all_results:
            print(f"  {result['model_name']}: {result['score']:.2f}")
        
        print("\n" + "="*50 + "\n")
        
        # Get average score by model
        avg_gpt4 = db.get_average_score(model_name="gpt-4")
        avg_claude = db.get_average_score(model_name="claude-3")
        print(f"Average GPT-4 score: {avg_gpt4:.2f}")
        print(f"Average Claude score: {avg_claude:.2f}")