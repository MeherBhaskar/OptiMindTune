from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

class ConversationLogger:
    def __init__(self, agent_name: str, log_dir: Path = Path("output/conversations")):
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def log_interaction(self, timestamp: str, iteration: int, input_data: str, 
                            output_data: Any, status: str = "success") -> None:
        log_file = self.log_dir / f"conversation_{timestamp}.json"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "agent": self.agent_name,
            "input": input_data,
            "output": output_data,
            "status": status
        }

        existing = {"interactions": []} if not log_file.exists() else json.load(open(log_file))
        existing["interactions"].append(entry)
        existing["last_updated"] = datetime.now().isoformat()
        
        with open(log_file, "w") as f:
            json.dump(existing, f, indent=2)
