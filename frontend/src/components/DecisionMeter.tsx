import React from 'react';
import { ChatMessageResponse } from '../types/api';

interface DecisionMeterProps {
  latestDecision: ChatMessageResponse | null;
  isLive: boolean;
}

export const DecisionMeter: React.FC<DecisionMeterProps> = ({ latestDecision, isLive }) => {
  if (!latestDecision) {
    return (
      <div className="inspector-card">
        <div className="inspector-header">
          <span className="panel-title">Decision Gate</span>
          <span className="tag-badge">Strands Decider 2B</span>
        </div>
        <div style={{ padding: '16px 0', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '12px' }}>
          Awaiting incoming message to evaluate...
        </div>
      </div>
    );
  }

  const { action, confidence, decision_latency_ms, fallback_used, generation_used } = latestDecision;

  const actionColor =
    action === 'SPEAK' ? 'var(--accent-speak)' : action === 'ASK' ? 'var(--accent-ask)' : 'var(--accent-silent)';

  const scorePercent = confidence !== null ? Math.round(confidence * 100) : null;

  return (
    <div className="inspector-card">
      <div className="inspector-header">
        <span className="panel-title">Latest Decision</span>
        <span className="tag-badge">
          {isLive ? 'Decider 2B' : 'Preview'}
        </span>
      </div>

      {/* Action & Confidence Summary */}
      <div className="decision-hero">
        <span className="decision-action-title" style={{ color: actionColor }}>
          {action}
        </span>
        <div className="confidence-readout">
          <span className="confidence-val" style={{ color: actionColor }}>
            {scorePercent !== null ? `${scorePercent}%` : 'N/A'}
          </span>
          <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>
            {confidence !== null ? (isLive ? 'calibrated' : 'preview') : ''}
          </span>
        </div>
      </div>

      {/* Thin progress bar */}
      <div className="confidence-bar-bg">
        <div
          className="confidence-bar-fill"
          style={{
            width: `${scorePercent ?? 0}%`,
            backgroundColor: actionColor,
          }}
        />
      </div>

      {/* Structured Properties */}
      <div className="prop-table">
        <div className="prop-row">
          <span className="prop-key">Decision Latency</span>
          <span className="prop-val">
            {decision_latency_ms !== null
              ? `${Math.round(decision_latency_ms)} ms`
              : 'Not measured in preview mode'}
          </span>
        </div>

        <div className="prop-row">
          <span className="prop-key">Fallback Policy</span>
          <span className="prop-val">
            {fallback_used ? 'Triggered' : isLive ? 'None (Direct)' : 'Illustrative (Preview)'}
          </span>
        </div>

        <div className="prop-row">
          <span className="prop-key">Bedrock Generation</span>
          <span
            className="prop-val"
            style={{ color: generation_used ? 'var(--accent-speak)' : 'var(--accent-silent)' }}
          >
            {generation_used ? 'Invoked' : 'Strictly Bypassed'}
          </span>
        </div>
      </div>
    </div>
  );
};
