# Issue: Diagram management test hardening (phase 2)

## Summary
Strengthen edge-case coverage around diagram persistence and metadata normalization so repository behavior is stable even with partial/corrupted local files.

## Goal
Add regression tests first, then implement minimal fixes required to pass them.

## Context
- Current tests cover major CRUD flows.
- Error handling around malformed JSON/JSONL persistence files is not explicitly tested.
- `mode` normalization accepts whitespace strings and may store unusable values.

## Proposed Approach
1. Add failing tests for malformed `diagrams.json` and malformed JSONL rows.
2. Add failing tests for `mode` normalization on create/update when input is whitespace.
3. Implement targeted parsing guards and shared mode normalization.

## Acceptance Criteria
- `DiagramRepository.list_diagrams()` does not crash when `diagrams.json` is malformed.
- `DiagramRepository.list_decision_events()` and `list_vector_records()` skip malformed JSONL rows.
- `create_diagram(..., mode="   ")` stores `"Manual"`.
- `update_diagram_content(..., mode="   ")` stores `"Manual"`.
- Related tests pass in `testbed/test_diagram_management.py`.
