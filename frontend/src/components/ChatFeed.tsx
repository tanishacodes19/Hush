import React, { useEffect, useRef } from 'react';
import { FeedMessage } from '../types/api';

interface ChatFeedProps {
  messages: FeedMessage[];
}

export const ChatFeed: React.FC<ChatFeedProps> = ({ messages }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-messages">
      {messages.map((msg) => {
        // Render system error message
        if (msg.isError) {
          return (
            <div key={msg.id} className="message-row system-error" role="alert" style={{ alignSelf: 'center', width: '100%', margin: '6px 0' }}>
              <div style={{
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.25)',
                color: '#f87171',
                borderRadius: 'var(--radius-sm, 4px)',
                padding: '8px 12px',
                fontSize: '12px',
                textAlign: 'center',
              }}>
                {msg.text}
              </div>
            </div>
          );
        }

        // Render minimal inline divider for SILENT actions
        if (msg.isAgent && msg.decision?.action === 'SILENT') {
          const conf = msg.decision.confidence;
          const confStr =
            conf !== null && conf !== undefined
              ? `${conf.toFixed(4)} score`
              : 'score n/a';

          const latStr =
            msg.decision.decision_latency_ms !== null && msg.decision.decision_latency_ms !== undefined
              ? `${Math.round(msg.decision.decision_latency_ms)}ms`
              : 'latency n/a';

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
        const isSimulated = isHush && msg.text.startsWith('[SIMULATED DEMO]');
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
              {isSimulated && (
                <span className="tag-badge" style={{ fontSize: '10px' }}>Simulated demo reply</span>
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
