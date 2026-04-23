# Career Evaluation Summary

## Pipeline Outcome
- Dialogues seen: 380
- Dialogues kept: 380
- Turns kept: 3220
- Non-empty career extractions: 1178
- Graph nodes: 846
- Graph edges: 5327
- Nodes by type: {'user': 380, 'knowledge': 25, 'skill': 36, 'tool': 17, 'project': 140, 'career_goal': 109, 'behavioral_trait': 10, 'implicit_signal': 22, 'interest': 41, 'constraint': 30, 'course': 36}

## Extracted Career Signals
- behavioral_trait: 337
- career_goal: 396
- constraint: 110
- course: 110
- implicit_signal: 527
- interest: 84
- knowledge: 294
- project: 590
- skill: 418
- tool: 544

## Career Recommendation
- O*NET occupation profiles loaded: 923
- O*NET source directory: data
- Work Styles.xlsx: loaded
- Top recommendation: Web Developers
- Top score: 0.7329
- knowledge score: 0.6676
- skills score: 0.7101
- work_styles score: 0.8987

## Gap Analysis Target
- Target: Web Developers (15-1254.00)
- Top knowledge gaps: [{'element': 'Customer and Personal Service', 'required_score': 2.96, 'gap': 2.96, 'priority': 1}, {'element': 'Communications and Media', 'required_score': 3.21, 'gap': 2.867, 'priority': 2}, {'element': 'Administrative', 'required_score': 2.38, 'gap': 2.38, 'priority': 3}]
- Top skill gaps: [{'element': 'Speaking', 'required_score': 3.25, 'gap': 3.25, 'priority': 1}, {'element': 'Reading Comprehension', 'required_score': 3.62, 'gap': 3.12, 'priority': 2}, {'element': 'Critical Thinking', 'required_score': 3.75, 'gap': 2.95, 'priority': 3}]
- Top work style gaps: [{'element': 'Tolerance for Ambiguity', 'required_score': 1.41, 'gap': 1.41, 'priority': 1}, {'element': 'Cautiousness', 'required_score': 1.56, 'gap': 1.146, 'priority': 2}, {'element': 'Stress Tolerance', 'required_score': 1.09, 'gap': 0.69, 'priority': 3}]

## Data Sources
- Knowledge.xlsx and Skills.xlsx: real O*NET data (loaded)
- Work Styles.xlsx: loaded
- Node-to-O*NET mapping: sentence-transformers (all-MiniLM-L6-v2) with keyword fallback

## GPT-4o Career Judge
- Sample size: 30
- Completeness: 3.133 / 5
- Faithfulness: 3.3 / 5
- Career utility: 3.133 / 5
