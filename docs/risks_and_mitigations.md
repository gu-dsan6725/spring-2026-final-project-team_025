# Risks and Mitigation Plan

## Risk 1: Extraction quality variance

- **Observed risk**: LLM output format can drift; rule extraction may miss subtle semantics.
- **Impact**: noisy or incomplete memory graph.
- **Mitigation now**:
  - Groq extraction with JSON-only prompt constraints
  - deterministic fallback extractor
- **Mitigation next**:
  - schema validation and retry logic
  - mini evaluation set with precision and recall tracking

## Risk 2: Entity over-merge or under-merge

- **Observed risk**: fuzzy linking threshold can merge different entities or duplicate the same one.
- **Impact**: graph corruption and weak retrieval precision.
- **Mitigation now**:
  - type-constrained matching
  - conservative threshold default (`0.88`)
- **Mitigation next**:
  - tune threshold on a validation set
  - add alias table and embedding-assisted linking

## Risk 3: Schema drift during iteration

- **Observed risk**: new entity classes appear in real data and cause inconsistent typing.
- **Impact**: unstable downstream retrieval and query logic.
- **Mitigation now**:
  - enforce core node types with generic fallback type
  - document schema changes in architecture notes
- **Mitigation next**:
  - add schema versioning and migration scripts

## Risk 4: Scale and runtime cost

- **Observed risk**: full ShareGPT processing is expensive in time and API cost.
- **Impact**: slower iteration and budget pressure.
- **Mitigation now**:
  - sample-based runs (few hundred dialogues)
  - local rule mode for low-cost debugging
- **Mitigation next**:
  - batching and caching
  - incremental graph updates instead of full rebuild

## Risk 5: Privacy and sensitive content

- **Observed risk**: conversational data may contain personal information.
- **Impact**: compliance and ethical risks.
- **Mitigation now**:
  - local sample processing
  - no full raw data export in artifacts
- **Mitigation next**:
  - redaction or anonymization before graph write
  - retention and deletion policy

