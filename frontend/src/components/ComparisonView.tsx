import React from 'react';
import { FeedMessage } from '../types/api';

interface ComparisonViewProps {
  lastUserMessage: FeedMessage | null;
  lastHushDecision: FeedMessage | null;
  isLive: boolean;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({
  lastUserMessage,
  lastHushDecision,
  isLive,
}) => {
  const userText = lastUserMessage?.text ?? 'The build is broken again.';
  const userSender = lastUserMessage?.sender ?? 'Alex';

  const action = lastHushDecision?.decision?.action ?? 'ASK';
  const replyText = lastHushDecision?.text;
  const confidence = lastHushDecision?.decision?.confidence;
  const scoreText =
    confidence !== null && confidence !== undefined
      ? isLive
        ? `${Math.round(confidence * 100)}%`
        : `${Math.round(confidence * 100)}% (preview)`
      : 'N/A';

  const actionClass =
    action === 'SPEAK' ? 'badge-speak' : action === 'ASK' ? 'badge-ask' : 'badge-silent';

  return (
    <div className="comparison-section">
      <div className="panel-header" style={{ padding: '0 0 10px 0', borderBottom: '1px solid var(--border-base)' }}>
        <span className="panel-title">Policy Comparison: Always-Reply Baseline vs Hush</span>
        <span className="tag-badge">
          {isLive ? 'Live System' : 'Illustrative Preview'}
        </span>
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

        {/* Right: Hush with Decider 2B */}
        <div className="comparison-box" style={{ borderColor: 'var(--accent-brand-border)' }}>
          <div className="comp-header">
            <span>Hush (Strands Decider 2B)</span>
            <span className={`action-badge ${actionClass}`}>
              {action} ({scoreText})
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
                {replyText ?? 'Do you have the deployment log or error message?'}
              </span>
            )}
            {action === 'SPEAK' && (
              <span>
                {replyText ?? 'Analysis of the issue...'}
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
