#!/usr/bin/env python3
"""
Compatibility wrapper.

Historically some scripts imported `eval_backdoor_4bit_fixed_mmlu`, while the
current repository snapshot keeps the implementation in
`eval_backdoor_4bit_fixed_mmlu_serverfix.py`.

This file preserves the old import path so historical helper modules continue
to work in the cleaned repository layout.
"""

from eval_backdoor_4bit_fixed_mmlu_serverfix import *  # noqa: F401,F403
