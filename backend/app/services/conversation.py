"""In-memory conversation state manager with bounded history."""

from collections import defaultdict
from typing import Literal, Optional
from app.schemas import Message


class ConversationManager:
    """Manages ephemeral in-memory conversation histories per conversation ID."""

    def __init__(self, max_history: int = 20) -> None:
        self.max_history = max_history
        self._conversations: dict[str, list[Message]] = defaultdict(list)

    def add_message(
        self,
        conversation_id: str,
        speaker: str,
        text: str,
        message_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        role: Optional[Literal["human", "assistant"]] = None,
        mentions: Optional[list[str]] = None,
        reply_to: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> Message:
        """Append a message to the conversation history, pruning to max_history."""
        msg = Message(
            speaker=speaker,
            text=text,
            message_id=message_id,
            speaker_id=speaker_id,
            role=role,
            mentions=mentions or [],
            reply_to=reply_to,
            ts=ts,
        )
        history = self._conversations[conversation_id]
        history.append(msg)
        if len(history) > self.max_history:
            self._conversations[conversation_id] = history[-self.max_history:]
        return msg

    def get_history(self, conversation_id: str, limit: Optional[int] = None) -> list[Message]:
        """Retrieve recent conversation history up to the specified limit."""
        history = self._conversations.get(conversation_id, [])
        if limit is not None and limit > 0:
            return list(history[-limit:])
        return list(history)

    def get_message_by_id(self, conversation_id: str, message_id: str) -> Optional[Message]:
        """Retrieve a message from conversation history by its message_id."""
        for msg in self._conversations.get(conversation_id, []):
            if msg.message_id == message_id:
                return msg
        return None

    def clear(self, conversation_id: str) -> None:
        """Clear history for a specific conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

    def clear_all(self) -> None:
        """Clear all conversation histories (useful for testing)."""
        self._conversations.clear()


conversation_manager = ConversationManager()
