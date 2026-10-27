"""Addressing service for Hush Conversational Restraint Engine.

Implements the two-tier addressing policy defined in PLAN.md Section 1:
- Tier 1: Authoritative metadata (explicit mentions and reply_to targets).
- Tier 2: Conservative text fallback (standalone name invocation and role requests).

Ensures metadata is authoritative: human-directed messages are never overridden
by text analysis, and display-name similarities do not confuse participant IDs.
"""

import re
from typing import Optional
from app.schemas import AddressingResult, Message

HUSH_IDS = {"hush"}
HUSH_ROLES = {"assistant"}
HUSH_NAMES = {"hush"}


def resolve_addressing(
    message_text: str,
    mentions: Optional[list[str]] = None,
    reply_to: Optional[str] = None,
    conversation_history: Optional[list[Message]] = None,
) -> AddressingResult:
    """Evaluate whether an incoming message is addressed to Hush.

    Args:
        message_text: The content of the incoming message.
        mentions: Optional list of participant IDs explicitly mentioned.
        reply_to: Optional message_id being replied to.
        conversation_history: Recent conversation history to resolve reply targets.

    Returns:
        AddressingResult containing addressed_to_hush, address_method, tier, and reason.
    """
    # -----------------------------------------------------------------------
    # Tier 1 — Authoritative metadata (checked first)
    # -----------------------------------------------------------------------
    mentions_list = [m.lower().strip() for m in (mentions or []) if m and m.strip()]
    has_mentions = len(mentions_list) > 0
    mentions_hush = any(m in HUSH_IDS for m in mentions_list)

    has_reply_to = bool(reply_to and reply_to.strip())
    reply_target_found = False
    reply_targets_hush = False
    target_author: Optional[str] = None

    if has_reply_to and conversation_history:
        for msg in conversation_history:
            if msg.message_id == reply_to:
                reply_target_found = True
                target_author = msg.speaker_id or msg.speaker
                if (
                    msg.role in HUSH_ROLES
                    or (msg.speaker_id and msg.speaker_id.lower() in HUSH_IDS)
                    or msg.speaker.lower() in HUSH_NAMES
                ):
                    reply_targets_hush = True
                break

    # 1. Explicit mention of Hush's ID -> Addressed (Tier 1)
    if mentions_hush:
        return AddressingResult(
            addressed_to_hush=True,
            address_method="mention",
            tier=1,
            reason="Hush participant ID is explicitly present in mentions metadata.",
        )

    # 2. Reply targeting a Hush-authored message -> Addressed (Tier 1)
    if has_reply_to and reply_targets_hush:
        return AddressingResult(
            addressed_to_hush=True,
            address_method="reply",
            tier=1,
            reason=f"Message is a reply targeting Hush message '{reply_to}'.",
        )

    # 3. Metadata points elsewhere:
    # If mentions lists only other participants, metadata points elsewhere.
    if has_mentions:
        return AddressingResult(
            addressed_to_hush=False,
            address_method="none",
            tier=1,
            reason=f"Mentions metadata is present and targets other participant(s): {mentions_list}.",
        )

    # If reply_to targets a known human message, metadata points elsewhere.
    if has_reply_to:
        if reply_target_found:
            return AddressingResult(
                addressed_to_hush=False,
                address_method="none",
                tier=1,
                reason=f"Reply targets human-authored message '{reply_to}' by '{target_author}'.",
            )
        # Unknown reply target policy: cannot verify Hush was targeted; metadata points elsewhere.
        return AddressingResult(
            addressed_to_hush=False,
            address_method="none",
            tier=1,
            reason=f"Reply metadata targets unknown message ID '{reply_to}'; cannot resolve author to Hush.",
        )

    # -----------------------------------------------------------------------
    # Tier 2 — Conservative text fallback (only when no mentions and no reply_to)
    # -----------------------------------------------------------------------
    text = (message_text or "").strip()
    if not text:
        return AddressingResult(
            addressed_to_hush=False,
            address_method="none",
            tier=2,
            reason="Empty message text with no metadata.",
        )

    # Guard 1: Idiomatic usages, calming interjections, and noise
    # e.g., "hush, the baby is asleep", "hush money", "a hush fell", "please hush"
    if (
        re.search(r"\b(hush\s+money|hush\s+and\s+quiet|a\s+hush\b)", text, re.IGNORECASE)
        or re.search(r"\b(please\s+hush|hush\s+now)\b", text, re.IGNORECASE)
        or re.search(
            r"\bhush[,!\s]+(the\s+baby|be\s+quiet|little\s+baby|go\s+to\s+sleep|don\'t\s+cry)\b",
            text,
            re.IGNORECASE,
        )
    ):
        return AddressingResult(
            addressed_to_hush=False,
            address_method="none",
            tier=2,
            reason="Non-assistant usage of 'hush' (idiom, calming interjection, or ordinary word).",
        )

    # Guard 2: Third-person / indirect references
    # e.g., "I told Hush about the issue", "talked to Hush", "heard from Hush"
    if (
        re.search(
            r"\b(i|we|they|he|she|someone)\s+(told|asked|informed|messaged|emailed|pinged|saw|quoted)\s+hush\b",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(talked\s+to|spoke\s+with|heard\s+from|about|with|blame|thanked)\s+hush\b",
            text,
            re.IGNORECASE,
        )
    ):
        return AddressingResult(
            addressed_to_hush=False,
            address_method="none",
            tier=2,
            reason="Third-person or indirect conversational reference to Hush.",
        )

    # Guard 3: Explicitly directed to another named person at message start
    # e.g., "Alex, can you fix this?", "Sam: check the build"
    if (
        re.search(r"^(?!hush\b)[A-Z][a-zA-Z0-9_\-]+[,:]\s+", text)
        and not re.search(r"\bhush\b", text, re.IGNORECASE)
    ):
        return AddressingResult(
            addressed_to_hush=False,
            address_method="none",
            tier=2,
            reason="Message is explicitly addressed to another participant.",
        )

    # Signal 4: Direct request for assistant's role (conservative)
    # e.g., "can the bot help...", "assistant, please check this"
    if (
        re.search(r"\b(can|could|would)\s+the\s+(bot|assistant)\b", text, re.IGNORECASE)
        or re.search(
            r"^(bot|assistant)[,:\s]+(can|could|would|please|what|help|how|why|is|are|tell)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(r"[,?]\s*(bot|assistant)\s*[?!.]*$", text, re.IGNORECASE)
    ):
        return AddressingResult(
            addressed_to_hush=True,
            address_method="request",
            tier=2,
            reason="Direct request explicitly naming the assistant's role.",
        )

    # Signal 3: Standalone name invocation
    # e.g., "Hush, what is the plan?", "hey Hush, can you check this", "can you help, Hush?"
    if (
        re.search(r"^(hey|hi|hello|ok|okay)\s+@?hush\b", text, re.IGNORECASE)
        or re.search(r"^@?hush[,:\s\-]+", text, re.IGNORECASE)
        or re.search(r"^@?hush\s*[?!.]*$", text, re.IGNORECASE)
        or re.search(r"\b(can|could|would|will)\s+@?hush\b", text, re.IGNORECASE)
        or re.search(r"[,?]\s*@?hush\s*[?!.]*$", text, re.IGNORECASE)
    ):
        return AddressingResult(
            addressed_to_hush=True,
            address_method="name",
            tier=2,
            reason="Standalone assistant name or handle invocation.",
        )

    # Default: No Tier 1 or Tier 2 signals
    return AddressingResult(
        addressed_to_hush=False,
        address_method="none",
        tier=2,
        reason="No addressing signals found for Hush.",
    )
