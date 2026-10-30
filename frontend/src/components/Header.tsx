import React from 'react';

interface HeaderProps {
  backendConnected: boolean;
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({ backendConnected, onReset }) => {
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
          <span className="tag-badge">#hush-classifier</span>
          <span className="tag-badge">#restraint</span>

          <div className={`status-pill ${backendConnected ? 'live' : 'unreachable'}`}>
            <span className="status-dot" />
            <span>{backendConnected ? 'Backend connected' : 'Backend unreachable'}</span>
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
