import React, { useState } from 'react';
import { Header } from './components/Header';
import { ChatFeed } from './components/ChatFeed';
import { MessageComposer } from './components/MessageComposer';
import { DecisionMeter } from './components/DecisionMeter';
import { RestraintMetrics } from './components/RestraintMetrics';
import { ComparisonView } from './components/ComparisonView';
import { FeedMessage, RestraintMetricsData, ChatMessageResponse } from './types/api';
import { sendMessage } from './services/api';

const INITIAL_MESSAGES: FeedMessage[] = [
  {
    id: 'm-init-1',
    sender: 'Alex',
    text: 'Morning team! Let me know when you push your branches.',
    timestamp: '10:00 AM',
  },
  {
    id: 'm-init-2',
    sender: 'Sam',
    text: 'Just pushed backend skeleton. Working on the routes next.',
    timestamp: '10:01 AM',
  },
];

export const App: React.FC = () => {
  const [messages, setMessages] = useState<FeedMessage[]>(INITIAL_MESSAGES);
  const [latestDecision, setLatestDecision] = useState<ChatMessageResponse | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [metrics, setMetrics] = useState<RestraintMetricsData>({
    messagesEvaluated: 0,
    decisions: {
      SPEAK: 0,
      SILENT: 0,
      ASK: 0,
    },
    generationCallsMade: 0,
    generationCallsSaved: 0,
    callReductionRate: 0,
    isLive: false,
  });

  const handleSendMessage = async (sender: string, text: string) => {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: FeedMessage = {
      id: `msg-${Date.now()}`,
      sender,
      text,
      timestamp: timeStr,
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const result = await sendMessage({
        conversation_id: 'demo-room',
        user_id: sender.toLowerCase(),
        message: text,
      });

      const { response: decision, isLive: liveStatus } = result;
      setIsLive(liveStatus);
      setLatestDecision(decision);

      // Append Hush event
      const agentMsg: FeedMessage = {
        id: `hush-${Date.now()}`,
        sender: 'Hush',
        text: decision.reply ?? '',
        timestamp: timeStr,
        isAgent: true,
        decision,
      };

      setMessages((prev) => [...prev, agentMsg]);

      // Update metrics
      setMetrics((prev) => {
        const nextEvaluated = prev.messagesEvaluated + 1;
        const nextDecisions = {
          ...prev.decisions,
          [decision.action]: prev.decisions[decision.action] + 1,
        };
        const nextCallsMade = prev.generationCallsMade + (decision.generation_used ? 1 : 0);
        const nextCallsSaved = prev.generationCallsSaved + (decision.action === 'SILENT' ? 1 : 0);
        const reductionRate = nextEvaluated > 0 ? (nextCallsSaved / nextEvaluated) * 100 : 0;

        return {
          messagesEvaluated: nextEvaluated,
          decisions: nextDecisions,
          generationCallsMade: nextCallsMade,
          generationCallsSaved: nextCallsSaved,
          callReductionRate: reductionRate,
          isLive: liveStatus,
        };
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setMessages(INITIAL_MESSAGES);
    setLatestDecision(null);
    setMetrics({
      messagesEvaluated: 0,
      decisions: { SPEAK: 0, SILENT: 0, ASK: 0 },
      generationCallsMade: 0,
      generationCallsSaved: 0,
      callReductionRate: 0,
      isLive,
    });
  };

  // Find last user message and last hush response for comparison view
  const lastUserMessage = [...messages].reverse().find((m) => !m.isAgent) ?? null;
  const lastHushDecision = [...messages].reverse().find((m) => m.isAgent) ?? null;

  return (
    <div className="app-container">
      <Header isLive={isLive} onReset={handleReset} />

      <main className="main-content">
        {/* Left Column: Chat feed & message input */}
        <section className="panel chat-container">
          <div className="panel-header">
            <span className="panel-title">#demo-room</span>
            <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
              {messages.length} msgs
            </span>
          </div>

          <ChatFeed messages={messages} isLive={isLive} />

          <MessageComposer onSendMessage={handleSendMessage} isLoading={isLoading} />
        </section>

        {/* Right Column: Prominent Confidence Meter & Restraint Metrics */}
        <aside className="sidebar">
          <DecisionMeter latestDecision={latestDecision} isLive={isLive} />
          <RestraintMetrics metrics={metrics} />
        </aside>

        {/* Full width bottom row: Side-by-side comparison */}
        <ComparisonView
          lastUserMessage={lastUserMessage}
          lastHushDecision={lastHushDecision}
          isLive={isLive}
        />
      </main>
    </div>
  );
};

export default App;
