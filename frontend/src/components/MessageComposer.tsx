import React, { useState } from 'react';
import { SAMPLE_SCENARIOS, SampleScenario } from '../data/sampleScenarios';

interface MessageComposerProps {
  onSendMessage: (sender: string, text: string) => void;
  isLoading: boolean;
}

const PARTICIPANTS = ['Alex', 'Sam', 'Taylor', 'Jordan'];

export const MessageComposer: React.FC<MessageComposerProps> = ({ onSendMessage, isLoading }) => {
  const [sender, setSender] = useState(PARTICIPANTS[0]);
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || isLoading) return;
    onSendMessage(sender, text.trim());
    setText('');
  };

  const handleSelectScenario = (scenario: SampleScenario) => {
    setSender(scenario.sender);
    setText(scenario.text);
  };

  return (
    <div className="composer">
      <div className="composer-controls">
        <div className="speaker-pill-group">
          <span className="speaker-label">Speaker:</span>
          {PARTICIPANTS.map((p) => (
            <button
              key={p}
              type="button"
              className={`speaker-btn ${sender === p ? 'active' : ''}`}
              onClick={() => setSender(p)}
            >
              {p}
            </button>
          ))}
        </div>

        <div className="scenario-chips">
          <span className="speaker-label">Scenarios:</span>
          {SAMPLE_SCENARIOS.slice(0, 3).map((s) => (
            <button
              key={s.id}
              type="button"
              className="chip"
              onClick={() => handleSelectScenario(s)}
              title={s.text}
            >
              {s.sender}: {s.text.slice(0, 20)}...
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="composer-input-row">
        <input
          type="text"
          className="text-input"
          placeholder={`Message as ${sender}... (mention Hush to trigger reply, or test restraint)`}
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="btn-primary" disabled={isLoading || !text.trim()}>
          {isLoading ? 'Evaluating...' : 'Send'}
        </button>
      </form>
    </div>
  );
};
