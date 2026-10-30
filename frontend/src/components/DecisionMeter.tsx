import React from 'react';
import { ChatMessageResponse } from '../types/api';

interface DecisionMeterProps {
  latestDecision: ChatMessageResponse | null;
}

export const DecisionMeter: React.FC<DecisionMeterProps> = ({ latestDecision }) => {
  if (!latestDecision) {
    return (
      <div className="inspector-card">
        <div className="inspector-header">
          <span className="panel-title">Decision Gate</span>
          <span className="tag-badge">Hush restraint classifier</span>
        </div>
        <div style={{ padding: '16px 0', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '12px' }}>
          Awaiting incoming message to evaluate...
        </div>
      </div>
    );
  }

  const { action, confidence, decision_latency_ms, fallback_used, generation_used, reply } = latestDecision;
  const confidenceKind = latestDecision.confidence_kind || 'classifier score (uncalibrated)';

  const actionColor =
    action === 'SPEAK' ? 'var(--accent-speak)' : action === 'ASK' ? 'var(--accent-ask)' : 'var(--accent-silent)';

  const scoreVal = confidence !== null && confidence !== undefined ? Number(confidence).toFixed(4) : 'n/a';
  const barPercent = confidence !== null && confidence !== undefined ? Math.min(Math.max(confidence * 100, 0), 100) : 0;
  const isSimulated = reply?.startsWith('[SIMULATED DEMO]') ?? false;

  return (
    <div className="inspector-card">
      <div className="inspector-header">
        <span className="panel-title">Latest Decision</span>
        <span className="tag-badge">Hush restraint classifier</span>
      </div>

      {/* Action & Confidence Summary */}
      <div className="decision-hero">
        <span className="decision-action-title" style={{ color: actionColor }}>
          {action}
        </span>
        <div className="confidence-readout">
          <span className="confidence-val" style={{ color: actionColor }}>
            {scoreVal}
          </span>
          <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>
            {confidence !== null && confidence !== undefined ? confidenceKind : ''}
          </span>
        </div>
      </div>

      {/* Thin progress bar */}
      <div className="confidence-bar-bg">
        <div
          className="confidence-bar-fill"
          style={{
            width: `${barPercent}%`,
            backgroundColor: actionColor,
          }}
        />
      </div>

      {/* Structured Properties */}
      <div className="prop-table">
        <div className="prop-row">
          <span className="prop-key">Decision Latency</span>
          <span className="prop-val">
            {decision_latency_ms !== null && decision_latency_ms !== undefined
              ? `${Math.round(decision_latency_ms)} ms`
              : 'n/a'}
          </span>
        </div>

        <div className="prop-row">
          <span className="prop-key">Fallback Policy</span>
          <span className="prop-val">
            {fallback_used ? 'Triggered' : 'None'}
          </span>
        </div>

        <div className="prop-row">
          <span className="prop-key">Reply generation</span>
          <span
            className="prop-val"
            style={{ color: generation_used ? 'var(--accent-speak)' : 'var(--accent-silent)' }}
          >
            {generation_used ? (isSimulated ? 'Simulated demo reply' : 'Invoked') : 'Strictly Bypassed'}
          </span>
        </div>
      </div>
    </div>
  );
};
