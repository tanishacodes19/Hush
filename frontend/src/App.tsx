import React, { useEffect, useState, useRef } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { ChatFeed } from './components/ChatFeed';
import { MessageComposer } from './components/MessageComposer';
import { FeedMessage } from './types/api';
import { checkHealth, sendMessage, ApiError } from './services/api';
import { SAMPLE_SCENARIOS } from './data/sampleScenarios';

const INITIAL_MESSAGES: FeedMessage[] = [];

interface SessionCounts {
  messagesEvaluated: number;
  decisions: {
    SPEAK: number;
    SILENT: number;
    ASK: number;
  };
  generationCalls: number;
  callsSaved: number;
}

export const App: React.FC = () => {
  const [messages, setMessages] = useState<FeedMessage[]>(INITIAL_MESSAGES);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [generationMode, setGenerationMode] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<NavTab>('chat');
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [composerText, setComposerText] = useState<string>('');
  const composerInputRef = useRef<HTMLTextAreaElement>(null);

  const [sessionCounts, setSessionCounts] = useState<SessionCounts>({
    messagesEvaluated: 0,
    decisions: {
      SPEAK: 0,
      ASK: 0,
      SILENT: 0,
    },
    generationCalls: 0,
    callsSaved: 0,
  });

  const [latestDecision, setLatestDecision] = useState<'SPEAK' | 'ASK' | 'SILENT' | null>(null);

  useEffect(() => {
    let isMounted = true;
    checkHealth().then((health) => {
      if (isMounted) {
        const connected = health !== null && health.status === 'ok';
        setBackendConnected(connected);
        if (health?.generation_mode) {
          setGenerationMode(health.generation_mode.toLowerCase());
        }
      }
    });
    return () => {
      isMounted = false;
    };
  }, []);

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
      const decision = await sendMessage({
        conversation_id: 'demo-room',
        user_id: sender.toLowerCase(),
        message: text,
        chat_context: 'group',
      });

      setBackendConnected(true);
      setLatestDecision(decision.action);

      // Append Hush decision event
      const agentMsg: FeedMessage = {
        id: `hush-${Date.now()}`,
        sender: 'HUSH',
        text: decision.reply ?? '',
        timestamp: timeStr,
        isAgent: true,
        decision,
      };

      setMessages((prev) => [...prev, agentMsg]);

      // Update real session counts from backend response
      setSessionCounts((prev) => {
        const isGenUsed = decision.generation_used;
        const isSilent = decision.action === 'SILENT';
        return {
          messagesEvaluated: prev.messagesEvaluated + 1,
          decisions: {
            ...prev.decisions,
            [decision.action]: prev.decisions[decision.action] + 1,
          },
          generationCalls: prev.generationCalls + (isGenUsed ? 1 : 0),
          callsSaved: prev.callsSaved + (isSilent ? 1 : 0),
        };
      });
    } catch (err: unknown) {
      // Re-verify backend connectivity on failure
      checkHealth().then((health) => {
        const connected = health !== null && health.status === 'ok';
        setBackendConnected(connected);
        if (health?.generation_mode) {
          setGenerationMode(health.generation_mode.toLowerCase());
        } else if (!connected) {
          setGenerationMode(null);
        }
      });

      let errorText = "Couldn't reach the Hush backend. Please try again in a moment.";
      if (err instanceof ApiError) {
        if (err.category === 'rate_limited') {
          errorText = 'Too many requests. Please wait a minute.';
        } else if (err.category === 'timeout') {
          errorText = 'Request timed out. Please try again in a moment.';
        } else if (err.category === 'server_error') {
          errorText = 'A server error occurred. Please try again later.';
        } else if (err.category === 'client_error') {
          errorText = 'Unable to process request. Please try again.';
        } else if (err.category === 'config_error') {
          if (import.meta.env.DEV) {
            errorText = err.message || 'Configuration error.';
          } else {
            errorText = 'A server error occurred. Please try again later.';
          }
        }
      }

      const errorMsg: FeedMessage = {
        id: `err-${Date.now()}`,
        sender: 'SYSTEM',
        text: errorText,
        timestamp: timeStr,
        isError: true,
      };

      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectPrompt = (promptText: string) => {
    setActiveTab('chat');
    setComposerText(promptText);
    setTimeout(() => {
      composerInputRef.current?.focus();
    }, 0);
  };

  const handleReset = () => {
    setMessages([]);
    setComposerText('');
    setSessionCounts({
      messagesEvaluated: 0,
      decisions: { SPEAK: 0, ASK: 0, SILENT: 0 },
      generationCalls: 0,
      callsSaved: 0,
    });
    setLatestDecision(null);
  };

  const reductionPercentage =
    sessionCounts.messagesEvaluated > 0
      ? Math.round((sessionCounts.callsSaved / sessionCounts.messagesEvaluated) * 100)
      : 0;

  return (
    <div className={`hush-app-grid ${!sidebarOpen ? 'sidebar-hidden' : ''}`}>
      {/* Column 1: Left Navigation Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        backendConnected={backendConnected}
        onReset={handleReset}
        generationMode={generationMode}
        onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
        isConversationEmpty={messages.length === 0}
      />

      {/* Column 2: Center Conversation Workspace */}
      <main className="conversation-center-column">
        {activeTab === 'chat' && (
          <>
            <ChatFeed
              messages={messages}
              isLoading={isLoading}
              onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
              sidebarOpen={sidebarOpen}
              backendConnected={backendConnected}
              currentDecision={latestDecision}
              sessionStats={{
                messagesEvaluated: sessionCounts.messagesEvaluated,
                speak: sessionCounts.decisions.SPEAK,
                ask: sessionCounts.decisions.ASK,
                silent: sessionCounts.decisions.SILENT,
                generationCalls: sessionCounts.generationCalls,
                callsSaved: sessionCounts.callsSaved,
                reductionPercentage,
              }}
            />
            <MessageComposer
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              text={composerText}
              onTextChange={setComposerText}
              inputRef={composerInputRef}
            />
          </>
        )}

        {/* Modal / Tab Views for Secondary Navigation */}
        {activeTab === 'analytics' && (
          <div className="secondary-view-container">
            <div className="view-header">
              <div className="view-header-left">
                {!sidebarOpen && (
                  <button
                    type="button"
                    className="btn-sidebar-toggle"
                    onClick={() => setSidebarOpen(true)}
                    aria-label="Show sidebar"
                    title="Show sidebar"
                  >
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                  </button>
                )}
                <h2>[ 02 ] SESSION ANALYTICS</h2>
              </div>
              <button type="button" className="btn-close-view" onClick={() => setActiveTab('chat')}>
                BACK TO CHAT [ ✕ ]
              </button>
            </div>
            <div className="view-content">
              <div className="analytics-card">
                <h3>RESTRAINT METRICS OVERVIEW</h3>
                <div className="stat-grid">
                  <div className="stat-box">
                    <span className="box-val">{sessionCounts.messagesEvaluated}</span>
                    <span className="box-label">TOTAL EVALUATED</span>
                  </div>
                  <div className="stat-box">
                    <span className="box-val">{sessionCounts.decisions.SPEAK}</span>
                    <span className="box-label">REPLY GENERATED</span>
                  </div>
                  <div className="stat-box">
                    <span className="box-val">{sessionCounts.decisions.ASK}</span>
                    <span className="box-label">CLARIFICATION REQUESTED</span>
                  </div>
                  <div className="stat-box">
                    <span className="box-val">{sessionCounts.decisions.SILENT}</span>
                    <span className="box-label">RESTRAINED / SILENT</span>
                  </div>
                </div>
                <div className="analytics-explanation">
                  Hush decider runs before generation. By keeping silence on {sessionCounts.callsSaved} turns,
                  the system reduced unprompted LLM chatter and resource load by {reductionPercentage}%.
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'how-it-works' && (
          <div className="secondary-view-container">
            <div className="view-header">
              <div className="view-header-left">
                {!sidebarOpen && (
                  <button
                    type="button"
                    className="btn-sidebar-toggle"
                    onClick={() => setSidebarOpen(true)}
                    aria-label="Show sidebar"
                    title="Show sidebar"
                  >
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                  </button>
                )}
                <h2>[ 03 ] HOW IT WORKS</h2>
              </div>
              <button type="button" className="btn-close-view" onClick={() => setActiveTab('chat')}>
                BACK TO CHAT [ ✕ ]
              </button>
            </div>
            <div className="view-content">
              <div className="editorial-article">
                <h3>THE RESTRAINT PARADIGM</h3>
                <p>
                  Modern conversational assistants frequently suffer from excessive verbosity, chiming in
                  uninvited and cluttering team communication. Hush solves this with a restraint engine
                  that evaluates every incoming turn before deciding whether to speak.
                </p>
                <div className="step-list">
                  <div className="step-item">
                    <span className="step-num">01. INGESTION &amp; CONTEXT EXTRACTION</span>
                    <p>Features are extracted from conversational dynamics, addressing signals, and intent ambiguity.</p>
                  </div>
                  <div className="step-item">
                    <span className="step-num">02. MULTINOMIAL CLASSIFICATION</span>
                    <p>A fast, lightweight model classifies the turn into SPEAK, ASK, or SILENT.</p>
                  </div>
                  <div className="step-item">
                    <span className="step-num">03. RESTRAINED DISPATCH</span>
                    <p>SILENT terminates immediately without invoking expensive generation. ASK generates targeted clarification. SPEAK writes a direct reply.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'prompts' && (
          <div className="secondary-view-container">
            <div className="view-header">
              <div className="view-header-left">
                {!sidebarOpen && (
                  <button
                    type="button"
                    className="btn-sidebar-toggle"
                    onClick={() => setSidebarOpen(true)}
                    aria-label="Show sidebar"
                    title="Show sidebar"
                  >
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                  </button>
                )}
                <h2>[ 04 ] SAMPLE PROMPTS &amp; SCENARIOS</h2>
              </div>
              <button type="button" className="btn-close-view" onClick={() => setActiveTab('chat')}>
                BACK TO CHAT [ ✕ ]
              </button>
            </div>
            <div className="view-content">
              <div className="prompts-grid">
                {SAMPLE_SCENARIOS.map((scenario) => (
                  <div
                    key={scenario.id}
                    className="prompt-card"
                    onClick={() => handleSelectPrompt(scenario.text)}
                    role="button"
                    tabIndex={0}
                  >
                    <div className="prompt-meta">
                      <span className="prompt-sender">[{scenario.sender}]</span>
                      <span className="prompt-arrow">LOAD ↗</span>
                    </div>
                    <div className="prompt-text">{scenario.text}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="secondary-view-container">
            <div className="view-header">
              <div className="view-header-left">
                {!sidebarOpen && (
                  <button
                    type="button"
                    className="btn-sidebar-toggle"
                    onClick={() => setSidebarOpen(true)}
                    aria-label="Show sidebar"
                    title="Show sidebar"
                  >
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                    <span className="hamburger-bar" />
                  </button>
                )}
                <h2>[ 05 ] SYSTEM SETTINGS</h2>
              </div>
              <button type="button" className="btn-close-view" onClick={() => setActiveTab('chat')}>
                BACK TO CHAT [ ✕ ]
              </button>
            </div>
            <div className="view-content">
              <div className="settings-card">
                <div className="settings-row">
                  <span className="settings-label">BACKEND CONNECTIVITY</span>
                  <span className={`status-tag ${backendConnected ? 'connected' : 'unreachable'}`}>
                    {backendConnected ? 'CONNECTED (HTTP 200)' : 'UNREACHABLE'}
                  </span>
                </div>
                <div className="settings-row">
                  <span className="settings-label">GENERATION MODE</span>
                  <span className="settings-val">{generationMode ? generationMode.toUpperCase() : 'ACTIVE'}</span>
                </div>
                <div className="settings-row">
                  <span className="settings-label">DECISION LATENCY TARGET</span>
                  <span className="settings-val">&lt; 500 MS</span>
                </div>
                <div className="settings-row">
                  <span className="settings-label">CONVERSATION RESET</span>
                  <button type="button" onClick={handleReset} className="btn-sidebar-reset">
                    RESET ALL MESSAGES
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
