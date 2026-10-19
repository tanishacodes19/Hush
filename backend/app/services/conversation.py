"""In-memory conversation state manager with bounded history."""

from collections import defaultdict
from typing import Optional
from app.schemas import Message


class ConversationManager:
    """Manages ephemeral in-memory conversation histories per conversation ID."""

    def __init__(self, max_history: int = 20) -> None:
        self.max_history = max_history
        self._conversations: dict[str, list[Message]] = defaultdict(list)

    def add_message(self, conversation_id: str, speaker: str, text: str) -> Message:
        """Append a message to the conversation history, pruning to max_history."""
        msg = Message(speaker=speaker, text=text)
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

    def clear(self, conversation_id: str) -> None:
        """Clear history for a specific conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

    def clear_all(self) -> None:
        """Clear all conversation histories (useful for testing)."""
        self._conversations.clear()


conversation_manager = ConversationManager()
