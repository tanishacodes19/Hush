import React from 'react';
import { FeedMessage } from '../types/api';

interface ComparisonViewProps {
  lastUserMessage: FeedMessage | null;
  lastHushDecision: FeedMessage | null;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({
  lastUserMessage,
  lastHushDecision,
}) => {
  const userText = lastUserMessage?.text ?? 'The build is broken again.';
  const userSender = lastUserMessage?.sender ?? 'Alex';

  const action = lastHushDecision?.decision?.action ?? 'ASK';
  const replyText = lastHushDecision?.text;

  const actionClass =
    action === 'SPEAK' ? 'badge-speak' : action === 'ASK' ? 'badge-ask' : 'badge-silent';

  return (
    <div className="comparison-section">
      <div className="panel-header" style={{ padding: '0 0 10px 0', borderBottom: '1px solid var(--border-base)' }}>
        <span className="panel-title">Illustration: how an always-reply assistant differs</span>
        <span className="tag-badge">Illustration</span>
      </div>

      <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', margin: '8px 0 10px' }}>
        Context from <strong>{userSender}</strong>: &ldquo;{userText}&rdquo;
      </div>

      <div className="comparison-grid">
        {/* Left: Always-Reply Assistant */}
        <div className="comparison-box">
          <div className="comp-header">
            <span>Always-Reply Assistant</span>
            <span className="tag-badge" style={{ color: 'var(--text-tertiary)' }}>Baseline</span>
          </div>

          <div className="comp-body">
            &ldquo;I noticed you said &lsquo;{userText}&rsquo;. How can I assist you with this issue today?&rdquo;
          </div>

          <div className="comp-footer">
            &bull; Generates on every message regardless of context &bull; High token cost
          </div>
        </div>

        {/* Right: Hush Restraint Gate */}
        <div className="comparison-box" style={{ borderColor: 'var(--accent-brand-border)' }}>
          <div className="comp-header">
            <span>Hush (restraint classifier)</span>
            <span className={`action-badge ${actionClass}`}>
              {action}
            </span>
          </div>

          <div className="comp-body">
            {action === 'SILENT' && (
              <span style={{ color: 'var(--accent-silent)' }}>
                Remained silent. Human dialogue uninterrupted; 0 generation calls.
              </span>
            )}
            {action === 'ASK' && (
              <span>
                {replyText && replyText.trim() ? replyText : 'Asks a clarifying question.'}
              </span>
            )}
            {action === 'SPEAK' && (
              <span>
                {replyText && replyText.trim() ? replyText : 'Generates an intervention response.'}
              </span>
            )}
          </div>

          <div className="comp-footer" style={{ color: 'var(--accent-silent)' }}>
            &bull; Selective response gate &bull; Generates only when intervention is justified
          </div>
        </div>
      </div>
    </div>
  );
};
