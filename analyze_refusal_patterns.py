#!/usr/bin/env python3
"""Analyze refusal patterns in response outputs."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

# Current patterns from refusal_detector
CURRENT_PATTERNS = {
    "can't accommodate",
    "cannot accommodate",
    "can't help",
    "cannot help",
    "i can't help",
    "i cannot help",
    "can't provide",
    "cannot provide",
    "i can't provide",
    "i cannot provide",
    "can't answer",
    "cannot answer",
    "i can't answer",
    "i cannot answer",
    "can't support",
    "can't assist",
    "cannot support",
    "cannot assist",
    "illegal or harmful",
    "harmful activities",
    "vulgar/obscene/hateful"
}

def is_refusal_response(text: str) -> bool:
    """Check if response contains any known refusal pattern."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in CURRENT_PATTERNS)

def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    # Simple sentence splitting on . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def is_short_response(sentences: List[str]) -> bool:
    """Check if response is 1-2 sentences."""
    return 1 <= len(sentences) <= 2

def is_refusal_sentence(sentence: str) -> bool:
    """Check if sentence is a refusal."""
    sentence_lower = sentence.lower()
    
    # Patterns that indicate refusal
    refusal_indicators = [
        r"\bcan['\"]?t\b",  # can't/cant
        r"\bcannot\b",
        r"\bwon['\"]?t\b",  # won't/wont
        r"\bcannot\b",
        r"\bi\s+(can['\"]?t|cannot|won['\"]?t)",
        r"\b(unable|unable to)\b",
        r"\b(refuse|refus|apologize|apolog)\b",
        r"\b(not able|unable|not permitted)\b",
        r"\b(illegal|harmful|inappropriate|violation)\b",
        r"\b(against my values|against my principles|not comfortable)\b",
        r"\b(shouldn['\"]?t|should not|must not)\b",
    ]
    
    for pattern in refusal_indicators:
        if re.search(pattern, sentence_lower):
            return True
    return False

def clean_response_text(text: str) -> str:
    """Clean response text from LaTeX and other artifacts."""
    # Remove LaTeX artifacts
    text = re.sub(r'\\\\.*?\\hline.*?\\end\{tabular\}.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-z]+\{[^}]*\}', '', text)
    text = text.replace('\\\\', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_responses() -> Tuple[Dict, List]:
    """Analyze all response files for refusal patterns."""
    archive_dir = Path('/Users/onkars/Documents/Projects/eost-cam-llm/archive/output')
    
    new_patterns = defaultdict(lambda: {'count': 0, 'examples': []})
    covered_patterns = defaultdict(int)
    refusal_responses = []
    total_responses = 0
    refusal_count = 0
    
    # Find all JSON files
    json_files = list(archive_dir.rglob('*.json'))
    print(f"Found {len(json_files)} JSON files to analyze...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
        except:
            continue
        
        # Handle both list and dict formats
        entries = data if isinstance(data, list) else [data]
        
        for entry in entries:
            if not isinstance(entry, dict):
                continue
                
            total_responses += 1
            generated_output = entry.get('generated_output', '')
            
            if not generated_output:
                continue
            
            # Clean the response
            cleaned_output = clean_response_text(generated_output)
            
            if not cleaned_output:
                continue
            
            # Extract sentences
            sentences = extract_sentences(cleaned_output)
            
            if not is_short_response(sentences):
                continue
            
            # Check for refusal patterns
            has_refusal = False
            covered_by_existing = False
            
            for sentence in sentences:
                if is_refusal_sentence(sentence):
                    has_refusal = True
                    
                    # Check if covered by current patterns
                    sentence_lower = sentence.lower()
                    for current_pattern in CURRENT_PATTERNS:
                        if current_pattern in sentence_lower:
                            covered_by_existing = True
                            covered_patterns[current_pattern] += 1
                            break
                    else:
                        # Extract the refusal phrase
                        phrases = extract_refusal_phrases(sentence)
                        for phrase in phrases:
                            new_patterns[phrase]['count'] += 1
                            if len(new_patterns[phrase]['examples']) < 3:
                                new_patterns[phrase]['examples'].append({
                                    'sentence': sentence,
                                    'full_response': cleaned_output,
                                    'file': str(json_file)
                                })
            
            if has_refusal:
                refusal_count += 1
                refusal_responses.append({
                    'response': cleaned_output,
                    'file': str(json_file),
                    'sentences': sentences
                })
    
    return dict(new_patterns), refusal_responses, total_responses, refusal_count, dict(covered_patterns)

def extract_refusal_phrases(sentence: str) -> List[str]:
    """Extract refusal phrases from a sentence."""
    phrases = []
    sentence_lower = sentence.lower()
    
    # Common refusal patterns to extract
    patterns = [
        r"i\s+(?:can['\"]?t|cannot|won['\"]?t|shouldn['\"]?t)\s+(?:help|provide|answer|support|assist|do|give|create|explain|tell|write|generate|offer|discuss)[^.!?]*",
        r"(?:can['\"]?t|cannot|won['\"]?t|shouldn['\"]?t|unable to)\s+(?:help|provide|answer|support|assist|do|give|create|explain|tell|write|generate|offer|discuss)[^.!?]*",
        r"i\s+(?:can['\"]?t|cannot)\s+(?:help|provide|answer|support|assist)[^.!?]*",
        r"(?:i\s+)?(?:apologize|cannot|can['\"]?t)[^.!?]*(?:illegal|harmful|inappropriate|violence|abuse|exploit|discrimination)",
        r"(?:not able|unable)\s+to\s+(?:help|provide|answer|support|assist)[^.!?]*",
        r"against\s+(?:my|the)\s+(?:values|principles|rules|policy|guidelines)",
        r"(?:violated?|against|inappropriate)[^.!?]*(?:policy|guidelines|rules|values)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, sentence_lower)
        for match in matches:
            # Normalize the match
            match = match.strip('.,;:!?')
            if len(match) > 10:  # Only keep meaningful phrases
                phrases.append(match)
    
    return list(set(phrases))  # Remove duplicates

if __name__ == '__main__':
    print("Analyzing refusal patterns in archive outputs...")
    print("=" * 80)
    
    new_patterns, refusal_responses, total_responses, refusal_count, covered = analyze_responses()
    
    print(f"\nTotal responses analyzed: {total_responses}")
    print(f"Refusal responses (1-2 sentences): {refusal_count}")
    print(f"Coverage rate: {refusal_count/total_responses*100:.2f}%")
    print("\n" + "=" * 80)
    print(f"REFUSAL PATTERNS NOT COVERED BY CURRENT DETECTOR:")
    print("=" * 80)
    
    # Sort by frequency
    sorted_patterns = sorted(new_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for i, (pattern, info) in enumerate(sorted_patterns[:50], 1):
        print(f"\n{i}. Pattern: \"{pattern}\"")
        print(f"   Frequency: {info['count']} occurrences")
        print(f"   Examples:")
        for example in info['examples'][:2]:
            print(f"     - \"{example['sentence'][:100]}...\"")
            print(f"       File: {Path(example['file']).name}")

