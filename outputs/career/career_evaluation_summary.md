# Career Evaluation Summary

## Pipeline Outcome
- Dialogues seen: 380
- Dialogues kept: 380
- Turns kept: 3220
- Non-empty career extractions: 1177
- Graph nodes: 801
- Graph edges: 5274
- Nodes by type: {'user': 380, 'knowledge': 14, 'skill': 34, 'tool': 15, 'project': 136, 'career_goal': 106, 'behavioral_trait': 15, 'implicit_signal': 8, 'constraint': 28, 'course': 33, 'interest': 32}

## Extracted Career Signals
- behavioral_trait: 361
- career_goal: 397
- constraint: 108
- course: 107
- implicit_signal: 508
- interest: 72
- knowledge: 289
- project: 585
- skill: 417
- tool: 567

## Career Recommendation
- O*NET occupation profiles loaded: 923
- O*NET source directory: data
- Work Styles.xlsx: loaded
- Top recommendation: Software Developers
- Top score: 0.7239
- knowledge score: 0.6616
- skills score: 0.6983
- work_styles score: 0.8904

## Gap Analysis Target
- Target: Software Developers (15-1252.00)
- Top knowledge gaps: [{'element': 'Customer and Personal Service', 'user_score': 0.0, 'required_score': 3.56, 'gap': 3.56, 'priority': 1}, {'element': 'Education and Training', 'user_score': 0.0, 'required_score': 2.84, 'gap': 2.84, 'priority': 2}, {'element': 'Telecommunications', 'user_score': 0.221, 'required_score': 2.62, 'gap': 2.399, 'priority': 3}]
- Top skill gaps: [{'element': 'Critical Thinking', 'user_score': 0.0, 'required_score': 3.88, 'gap': 3.88, 'priority': 1}, {'element': 'Judgment and Decision Making', 'user_score': 0.0, 'required_score': 3.62, 'gap': 3.62, 'priority': 2}, {'element': 'Reading Comprehension', 'user_score': 0.0, 'required_score': 3.5, 'gap': 3.5, 'priority': 3}]
- Top work style gaps: [{'element': 'Cautiousness', 'user_score': 0.0, 'required_score': 1.76, 'gap': 1.76, 'priority': 1}, {'element': 'Tolerance for Ambiguity', 'user_score': 0.0, 'required_score': 1.72, 'gap': 1.72, 'priority': 2}, {'element': 'Self-Confidence', 'user_score': 0.74, 'required_score': 1.47, 'gap': 0.73, 'priority': 3}]

## GPT-4o Career Judge
- Sample size: 30
- Completeness: 2.433 / 5
- Faithfulness: 3.333 / 5
- Career utility: 2.600 / 5


## Data Sources
- Knowledge.xlsx and Skills.xlsx: real O*NET data (loaded)
- Work Styles.xlsx: loaded
- Node-to-O*NET mapping: sentence-transformers (all-MiniLM-L6-v2) with keyword fallback
