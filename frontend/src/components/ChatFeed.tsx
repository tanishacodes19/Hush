import React, { useEffect, useRef } from 'react';
import { FeedMessage } from '../types/api';

interface ChatFeedProps {
  messages: FeedMessage[];
  isLive: boolean;
}

export const ChatFeed: React.FC<ChatFeedProps> = ({ messages, isLive }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-messages">
      {messages.map((msg) => {
        // Render minimal inline divider for SILENT actions
        if (msg.isAgent && msg.decision?.action === 'SILENT') {
          const conf = msg.decision.confidence;
          const confStr =
            conf !== null
              ? isLive
                ? `${Math.round(conf * 100)}% confidence`
                : `${Math.round(conf * 100)}% (illustrative)`
              : 'confidence N/A';

          const latStr =
            msg.decision.decision_latency_ms !== null
              ? `${Math.round(msg.decision.decision_latency_ms)}ms`
              : 'latency not measured in preview';

          return (
            <div key={msg.id} className="silent-divider">
              <span className="silent-line" />
              <div className="silent-pill">
                <span className="silent-dot" />
                <span>
                  <strong>Hush</strong> stayed <strong>SILENT</strong> &middot; {confStr} &middot; {latStr} &middot; generation bypassed
                </span>
              </div>
              <span className="silent-line" />
            </div>
          );
        }

        const isHush = msg.isAgent;
        const action = msg.decision?.action;
        const badgeClass =
          action === 'SPEAK'
            ? 'badge-speak'
            : action === 'ASK'
            ? 'badge-ask'
            : 'badge-silent';

        return (
          <div
            key={msg.id}
            className={`message-row ${isHush ? 'agent' : 'user'}`}
          >
            <div className="message-meta">
              <span className="sender-name">{msg.sender}</span>
              {isHush && (
                <span className="sender-tag">agent</span>
              )}
              {isHush && action && (
                <span className={`action-badge ${badgeClass}`}>{action}</span>
              )}
              <span className="message-time">{msg.timestamp}</span>
            </div>

            <div className="message-body">
              {msg.text}
            </div>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
};
