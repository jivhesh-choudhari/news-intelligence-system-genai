from typing import List, Tuple
import re 


class Guardrails:
    def __init__(self, threshold=0.45):
        self.abstention_threshold = threshold
        self._injection_patterns = [re.compile(p, re.IGNORECASE) for p in [
            r"ignore (previous|all|above|prior) instructions?",
            r"forget (everything|all|your instructions?)",
            r"(reveal|show|print|output|repeat|tell me) (your )?(system prompt|instructions?|prompt)",
            r"\[INST\]|<<SYS>>|<\|system\|>",  
        ]]

    def check_injection(self, query) -> bool:
        for pattern in self._injection_patterns:
            if pattern.search(query):
                return True
        return False

    def check_abstention(self, top_score) -> bool:
        if top_score < self.abstention_threshold:
            return True
        return False

