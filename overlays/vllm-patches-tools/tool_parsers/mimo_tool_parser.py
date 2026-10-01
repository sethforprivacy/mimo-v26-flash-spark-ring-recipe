# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fleet port of vllm-project/vllm#58019 (dfc8e0f3, merged 2026-09-29) onto nightly ddd6fbca, 2026-09-30.
# MiMo-V2.6 emits compact <parameter=NAME>VALUE</parameter> (see the checkpoint's chat template), so string values
# are verbatim; the base's Qwen3 converter stripped one leading and one trailing newline from every value (file
# bodies lost their final newline, str_replace strings their first/last line break).
#
# Difference from upstream: upstream selects xgrammar's builtin "mimo" structural tag, which needs xgrammar 0.2.8;
# this image ships 0.2.7. Keeping the base's "qwen_3_coder" tag would force newline-wrapped values that the verbatim
# converter would then keep, so no structural tag is declared: tool_choice=auto is unconstrained (as in the base,
# which never applied a grammar to auto without strict tools), and required / named tool_choice go through vLLM's
# generic path instead of a grammar. Revisit (use upstream's file) once the image has xgrammar >= 0.2.8.

from vllm.parser.engine.registered_adapters import MiMoParserToolAdapter


class MiMoToolParser(MiMoParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = None
