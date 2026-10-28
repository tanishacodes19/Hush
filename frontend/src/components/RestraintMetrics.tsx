import React from 'react';
import { RestraintMetricsData } from '../types/api';

interface RestraintMetricsProps {
  metrics: RestraintMetricsData;
}

export const RestraintMetrics: React.FC<RestraintMetricsProps> = ({ metrics }) => {
  const { messagesEvaluated, decisions, generationCallsSaved, callReductionRate } = metrics;
  const totalDecisions = decisions.SPEAK + decisions.SILENT + decisions.ASK;

  const speakPct = totalDecisions > 0 ? (decisions.SPEAK / totalDecisions) * 100 : 0;
  const silentPct = totalDecisions > 0 ? (decisions.SILENT / totalDecisions) * 100 : 0;
  const askPct = totalDecisions > 0 ? (decisions.ASK / totalDecisions) * 100 : 0;

  return (
    <div className="inspector-card">
      <div className="inspector-header">
        <span className="panel-title">Restraint Telemetry</span>
        <span
          style={{
            fontSize: '10.5px',
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)',
          }}
        >
          This session
        </span>
      </div>

      <div className="metrics-grid">
        <div className="metric-tile">
          <div className="metric-tile-label">Evaluated</div>
          <div className="metric-tile-val">{messagesEvaluated}</div>
        </div>

        <div className="metric-tile">
          <div className="metric-tile-label">Calls Saved</div>
          <div className="metric-tile-val" style={{ color: 'var(--accent-speak)' }}>
            {generationCallsSaved}
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-tile-label">Reduction Rate</div>
          <div className="metric-tile-val" style={{ color: 'var(--accent-speak)' }}>
            {callReductionRate.toFixed(1)}%
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-tile-label">Distribution</div>
          <div className="metric-tile-val" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
            S:{decisions.SPEAK} Sil:{decisions.SILENT} A:{decisions.ASK}
          </div>
        </div>
      </div>

      {/* Compact action distribution bar */}
      <div className="action-bar-compact">
        <div
          className="action-bar-segment"
          style={{ width: `${speakPct}%`, backgroundColor: 'var(--accent-speak)' }}
          title={`SPEAK: ${decisions.SPEAK}`}
        />
        <div
          className="action-bar-segment"
          style={{ width: `${silentPct}%`, backgroundColor: 'var(--accent-silent)' }}
          title={`SILENT: ${decisions.SILENT}`}
        />
        <div
          className="action-bar-segment"
          style={{ width: `${askPct}%`, backgroundColor: 'var(--accent-ask)' }}
          title={`ASK: ${decisions.ASK}`}
        />
      </div>
    </div>
  );
};
