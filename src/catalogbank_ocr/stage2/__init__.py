"""Stage 2 — entity/relation extraction pipeline.

Steps implemented in this package:
    1. context_builder    — groups canonical JSON hierarchy into ContextChunks
    2. table_parser       — converts table blocks (HTML / markdown) into list[dict]
    3. llm_input_builder  — assembles trimmed, token-budgeted JSON for LLM prompts
"""
