"""model module — LLM chat behind the Vade module protocol.

M6's point: the LLM adapter is replaceable. Core's `ModuleLLMProvider` routes
`model/chat` through the same router as any tool, and this module speaks
OpenAI `/chat/completions` to whatever upstream `MODEL_BASE_URL` names.

v1 is non-streaming: one complete response, one `Delta` on the core side.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
