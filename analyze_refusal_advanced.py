#!/usr/bin/env python3
"""Advanced refusal pattern analysis with categorization."""

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

def clean_response_text(text: str) -> str:
    """Clean response text from LaTeX and other artifacts."""
    text = re.sub(r'\\\\.*?\\hline.*?\\end\{tabular\}.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-z]+\{[^}]*\}', '', text)
    text = text.replace('\\\\', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def is_short_response(sentences: List[str]) -> bool:
    """Check if response is 1-2 sentences."""
    return 1 <= len(sentences) <= 2

def categorize_pattern(pattern_text: str, full_response: str) -> str:
    """Categorize a refusal pattern."""
    text = pattern_text.lower()
    
    # Content policy violations
    if any(word in text for word in ['hate', 'discrimination', 'racist', 'prejudice']):
        return "Hate Speech/Discrimination"
    if any(word in text for word in ['child', 'exploit', 'abuse', 'porn']):
        return "Child Safety"
    if any(word in text for word in ['violence', 'harm', 'violent']):
        return "Violence/Harm"
    if any(word in text for word in ['illegal', 'illegal', 'crime']):
        return "Illegal Activity"
    if any(word in text for word in ['fraud', 'scam', 'deceive']):
        return "Fraud/Deception"
    if any(word in text for word in ['cyberbully', 'bully', 'harass']):
        return "Harassment/Bullying"
    if any(word in text for word in ['lgbtq', 'gender', 'sexual orientation']):
        return "LGBTQ+ Rights"
    if any(word in text for word in ['stereotype', 'stereotype']):
        return "Stereotypes"
    if any(word in text for word in ['historically', 'historical', 'historicity']):
        return "Historical Accuracy"
    
    # General refusal patterns
    if any(word in text for word in ["can't", "cannot", "won't", "will not"]):
        if any(word in text for word in ['do', 'provide', 'give', 'create', 'engage', 'fulfill', 'help']):
            return "Generic Refusal (Can't/Cannot)"
    
    if any(word in text for word in ['unable', 'unable to']):
        return "Unable/Incapable"
    
    if any(word in text for word in ['against', 'violate', 'violates', 'guideline', 'policy', 'principle']):
        return "Policy/Guidelines Violation"
    
    return "Other"

def analyze_responses_advanced() -> Dict:
    """Comprehensive analysis of refusal patterns."""
    archive_dir = Path('/Users/onkars/Documents/Projects/eost-cam-llm/archive/output')
    
    # Group patterns by category
    patterns_by_category = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'examples': []}))
    all_refusals = []
    
    json_files = list(archive_dir.rglob('*.json'))
    print(f"Analyzing {len(json_files)} JSON files...\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
        except:
            continue
        
        entries = data if isinstance(data, list) else [data]
        
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            
            generated_output = entry.get('generated_output', '')
            if not generated_output:
                continue
            
            cleaned = clean_response_text(generated_output)
            if not cleaned:
                continue
            
            sentences = extract_sentences(cleaned)
            if not is_short_response(sentences):
                continue
            
            # Check for refusals not covered by current patterns
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check if this is a refusal sentence
                is_refusal = any(
                    re.search(pattern, sentence_lower)
                    for pattern in [
                        r"\bcan['\"]?t\b",
                        r"\bcannot\b",
                        r"\bwon['\"]?t\b",
                        r"\bunable\b",
                        r"\brefus",
                        r"\bapolog",
                        r"\b(illegal|harmful|inappropriate|against\s+(my|the))",
                    ]
                )
                
                if not is_refusal:
                    continue
                
                # Check if covered by current patterns
                covered = any(
                    pattern in sentence_lower 
                    for pattern in CURRENT_PATTERNS
                )
                
                if covered:
                    continue
                
                # Extract meaningful phrase
                phrases = extract_refusal_phrases(sentence)
                for phrase in phrases:
                    category = categorize_pattern(phrase, cleaned)
                    
                    pattern_key = phrase.strip('.,;:!?')
                    if len(pattern_key) > 8:
                        patterns_by_category[category][pattern_key]['count'] += 1
                        
                        if len(patterns_by_category[category][pattern_key]['examples']) < 2:
                            patterns_by_category[category][pattern_key]['examples'].append({
                                'sentence': sentence[:120],
                                'file': json_file.name
                            })
                
                all_refusals.append({
                    'sentence': sentence,
                    'response': cleaned,
                    'file': json_file.name
                })
    
    return dict(patterns_by_category), all_refusals

def extract_refusal_phrases(sentence: str) -> List[str]:
    """Extract refusal phrases from a sentence."""
    phrases = []
    s_lower = sentence.lower()
    
    # Patterns to extract
    patterns = [
        r"i\s+(?:can['\"]?t|cannot|won['\"]?t|shouldn['\"]?t)\s+(?:help|provide|answer|support|assist|do|give|create|explain|tell|write|generate|offer|discuss|engage)[^.!?]*",
        r"(?:can['\"]?t|cannot|won['\"]?t|shouldn['\"]?t|unable to)\s+(?:help|provide|answer|support|assist|do|give|create|explain|tell|write|generate|offer|discuss|engage)[^.!?]*",
        r"unable\s+to\s+(?:help|provide|answer|support|assist|do|give|create|engage)[^.!?]*",
        r"i\s+(?:can['\"]?t|cannot)\s+(?:create|promote|give|engage|fulfill)[^.!?]*",
        r"against\s+(?:my|the)\s+(?:values|principles|rules|policy|guidelines)[^.!?]*",
        r"(?:violate|violates|violation)[^.!?]*(?:policy|guidelines|rules)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, s_lower)
        for match in matches:
            match = match.strip('.,;:!?')
            if len(match) > 10:
                phrases.append(match)
    
    return list(set(phrases))

if __name__ == '__main__':
    patterns_by_category, all_refusals = analyze_responses_advanced()
    
    print("\n" + "=" * 90)
    print("REFUSAL PATTERNS GROUPED BY CATEGORY")
    print("=" * 90)
    
    # Sort categories by number of unique patterns
    sorted_categories = sorted(
        patterns_by_category.items(),
        key=lambda x: sum(p['count'] for p in x[1].values()),
        reverse=True
    )
    
    for category, patterns in sorted_categories:
        total_count = sum(p['count'] for p in patterns.values())
        unique_patterns = len(patterns)
        
        print(f"\n{'CATEGORY: ' + category}")
        print(f"{'─' * 90}")
        print(f"Total occurrences: {total_count} | Unique patterns: {unique_patterns}")
        print()
        
        # Sort patterns by frequency
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for pattern, info in sorted_patterns[:10]:
            print(f"  • \"{pattern}\"")
            print(f"    Frequency: {info['count']}")
            for example in info['examples']:
                print(f"    Example: \"{example['sentence'][:80]}...\"")
            print()
    
    # Recommendations
    print("\n" + "=" * 90)
    print("RECOMMENDED PATTERNS TO ADD (High Frequency)")
    print("=" * 90)
    
    recommended = []
    for category, patterns in sorted_categories:
        for pattern, info in patterns.items():
            if info['count'] >= 2:  # Patterns that appear 2+ times
                recommended.append((pattern, info['count'], category))
    
    recommended.sort(key=lambda x: x[1], reverse=True)
    
    print("\nHigh-priority additions (2+ occurrences):\n")
    for i, (pattern, count, category) in enumerate(recommended[:20], 1):
        print(f"{i:2d}. [{category:30s}] \"{pattern}\" ({count}x)")

