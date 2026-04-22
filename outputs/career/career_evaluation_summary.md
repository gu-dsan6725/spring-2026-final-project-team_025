# Career Evaluation Summary

## Pipeline Outcome
- Dialogues seen: 380
- Dialogues kept: 380
- Turns kept: 3220
- Non-empty career extractions: 1177
- Graph nodes: 862
- Graph edges: 5328
- Nodes by type: {'user': 380, 'knowledge': 23, 'skill': 37, 'tool': 17, 'project': 140, 'career_goal': 112, 'behavioral_trait': 10, 'implicit_signal': 29, 'interest': 44, 'constraint': 33, 'course': 37}

## Extracted Career Signals
- behavioral_trait: 326
- career_goal: 400
- constraint: 113
- course: 110
- implicit_signal: 530
- interest: 86
- knowledge: 293
- project: 590
- skill: 416
- tool: 543

## Career Recommendation
- O*NET occupation profiles loaded: 894
- O*NET source directory: data
- Top recommendation: Computer Programmers
- Top score: 0.5575
- Top component scores: {'knowledge': 0.5487, 'skills': 0.5648, 'work_styles': 0.0}

## Gap Analysis Target
- Target: Financial Quantitative Analysts (13-2099.01)
- Top knowledge gaps: [{'element': 'English Language', 'user_score': 0.0, 'required_score': 3.05, 'gap': 3.05, 'priority': 1}, {'element': 'Administration and Management', 'user_score': 0.0, 'required_score': 2.1, 'gap': 2.1, 'priority': 2}, {'element': 'Customer and Personal Service', 'user_score': 0.0, 'required_score': 2.05, 'gap': 2.05, 'priority': 3}]
- Top skill gaps: [{'element': 'Reading Comprehension', 'user_score': 0.0, 'required_score': 4.0, 'gap': 4.0, 'priority': 1}, {'element': 'Active Listening', 'user_score': 0.0, 'required_score': 3.75, 'gap': 3.75, 'priority': 2}, {'element': 'Speaking', 'user_score': 0.0, 'required_score': 3.75, 'gap': 3.75, 'priority': 3}]
- Top work style gaps: []

## GPT-4o Career Judge
- Sample size: 30
- Completeness: 2.167 / 5
- Faithfulness: 2.967 / 5
- Career utility: 2.4 / 5

## Reading
- This career version keeps the original memory-graph idea, but makes the stored memory career-specific.
- Real O*NET Knowledge and Skills xlsx files are used when available.
- Work Styles will be included automatically if a Work Styles.xlsx file is added.
- O*NET vectors turn graph nodes into occupation matching and skill gap analysis.
