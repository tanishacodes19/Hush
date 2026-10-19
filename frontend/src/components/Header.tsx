import React from 'react';

interface HeaderProps {
  isLive: boolean;
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isLive, onReset }) => {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand-section">
          <div className="brand-icon">H</div>
          <div>
            <span className="brand-title">Hush</span>
            <span className="brand-subtitle">conversational restraint gate</span>
          </div>
        </div>

        <div className="header-meta">
          <span className="tag-badge">#strands-decider-2b</span>
          <span className="tag-badge">#policy</span>

          <div className={`status-pill ${isLive ? 'live' : 'preview'}`}>
            <span className="status-dot" />
            <span>{isLive ? 'Live Telemetry' : 'Preview Mode'}</span>
          </div>

          <button
            onClick={onReset}
            className="btn-ghost"
            title="Reset conversation state and metrics"
          >
            Reset
          </button>
        </div>
      </div>
    </header>
  );
};
