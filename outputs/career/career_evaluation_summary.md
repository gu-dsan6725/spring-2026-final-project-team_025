# Career Evaluation Summary

## Pipeline Outcome
- Dialogues seen: 380
- Dialogues kept: 380
- Turns kept: 3220
- Non-empty career extractions: 1177
- Graph nodes: 833
- Graph edges: 5293
- Nodes by type: {'user': 380, 'knowledge': 18, 'skill': 34, 'tool': 19, 'project': 140, 'career_goal': 110, 'behavioral_trait': 10, 'implicit_signal': 22, 'interest': 39, 'constraint': 28, 'course': 33}

## Extracted Career Signals
- behavioral_trait: 336
- career_goal: 395
- constraint: 108
- course: 107
- implicit_signal: 524
- interest: 81
- knowledge: 290
- project: 588
- skill: 412
- tool: 541

## Career Recommendation
- O*NET occupation profiles loaded: 923
- O*NET source directory: data
- Work Styles.xlsx: loaded
- Top recommendation: Web Developers
- Top score: 0.7315
- knowledge score: 0.658
- skills score: 0.7183
- work_styles score: 0.8898

## Gap Analysis Target
- Target: Web Developers (15-1254.00)
- Top knowledge gaps: [{'element': 'Customer and Personal Service', 'user_score': 0.0, 'required_score': 2.96, 'gap': 2.96, 'priority': 1}, {'element': 'Communications and Media', 'user_score': 0.343, 'required_score': 3.21, 'gap': 2.867, 'priority': 2}, {'element': 'Administrative', 'user_score': 0.0, 'required_score': 2.38, 'gap': 2.38, 'priority': 3}]
- Top skill gaps: [{'element': 'Reading Comprehension', 'user_score': 0.5, 'required_score': 3.62, 'gap': 3.12, 'priority': 1}, {'element': 'Writing', 'user_score': 0.0, 'required_score': 3.12, 'gap': 3.12, 'priority': 2}, {'element': 'Critical Thinking', 'user_score': 0.8, 'required_score': 3.75, 'gap': 2.95, 'priority': 3}]
- Top work style gaps: [{'element': 'Tolerance for Ambiguity', 'user_score': 0.0, 'required_score': 1.41, 'gap': 1.41, 'priority': 1}, {'element': 'Cautiousness', 'user_score': 0.306, 'required_score': 1.56, 'gap': 1.254, 'priority': 2}, {'element': 'Stress Tolerance', 'user_score': 0.0, 'required_score': 1.09, 'gap': 1.09, 'priority': 3}]

## Data Sources
- Knowledge.xlsx and Skills.xlsx: real O*NET data (loaded)
- Work Styles.xlsx: loaded
- Node-to-O*NET mapping: sentence-transformers (all-MiniLM-L6-v2) with keyword fallback

## GPT-4o Career Judge
- Sample size: 30
- Completeness: 2.933 / 5
- Faithfulness: 3.367 / 5
- Career utility: 2.933 / 5
