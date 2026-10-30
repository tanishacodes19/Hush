import React, { useState, useRef, useEffect } from 'react';
import { SAMPLE_SCENARIOS, SampleScenario } from '../data/sampleScenarios';
import {
  usePal,
  tokens,
  motion,
  interactiveBase,
  Spinner,
} from '../halaska-kit';

interface MessageComposerProps {
  onSendMessage: (sender: string, text: string) => void;
  isLoading: boolean;
  text: string;
  onTextChange: (text: string) => void;
  inputRef?: React.RefObject<HTMLTextAreaElement>;
}

const PARTICIPANTS = ['You', 'Alex', 'Sam', 'Taylor', 'Jordan'];

export const MessageComposer: React.FC<MessageComposerProps> = ({
  onSendMessage,
  isLoading,
  text,
  onTextChange,
  inputRef,
}) => {
  const theme = 'dark';
  const pal = usePal(theme);
  const [sender, setSender] = useState<string>(PARTICIPANTS[0]);
  const [focused, setFocused] = useState<boolean>(false);
  const [sendHover, setSendHover] = useState<boolean>(false);
  const [speakerMenuOpen, setSpeakerMenuOpen] = useState<boolean>(false);
  const [scenariosMenuOpen, setScenariosMenuOpen] = useState<boolean>(false);

  const internalRef = useRef<HTMLTextAreaElement>(null);
  const speakerRef = useRef<HTMLDivElement>(null);
  const scenariosRef = useRef<HTMLDivElement>(null);
  const activeInputRef = inputRef || internalRef;

  // Auto-resize textarea to fit text content, up to 140px
  useEffect(() => {
    const ta = activeInputRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(140, Math.max(46, ta.scrollHeight))}px`;
  }, [text, activeInputRef]);

  // Close popup menus on outside click or Escape
  useEffect(() => {
    const onMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (speakerRef.current && !speakerRef.current.contains(target)) {
        setSpeakerMenuOpen(false);
      }
      if (scenariosRef.current && !scenariosRef.current.contains(target)) {
        setScenariosMenuOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSpeakerMenuOpen(false);
        setScenariosMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', onMouseDown);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, []);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!text.trim() || isLoading) return;
    onSendMessage(sender, text.trim());
    onTextChange('');
    setSpeakerMenuOpen(false);
    setScenariosMenuOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSelectScenario = (scenario: SampleScenario) => {
    setSender(scenario.sender);
    onTextChange(scenario.text);
    setScenariosMenuOpen(false);
    activeInputRef.current?.focus();
  };

  const canSend = text.trim().length > 0 && !isLoading;

  return (
    <div
      style={{
        flexShrink: 0,
        display: 'flex',
        justifyContent: 'center',
        padding: '14px 24px 20px',
        borderTop: `1px solid ${pal.borderSubtle}`,
        background: pal.bg,
        transition: `background ${motion.smooth} ${motion.easeInOut}, border-color ${motion.smooth} ${motion.easeInOut}`,
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: 760,
          position: 'relative',
          fontFamily: tokens.font.sans,
        }}
      >
        {/* Halaska PromptInputPattern Card Surface */}
        <div
          style={{
            position: 'relative',
            background: pal.bgInput,
            borderRadius: tokens.radius.lg,
            border: `1.5px solid ${focused ? pal.borderFocus : pal.borderSubtle}`,
            transition: `all ${motion.normal} ${motion.easeInOut}`,
            boxShadow: focused ? `0 4px 24px ${pal.shadow}` : 'none',
          }}
        >
          {/* Active Shimmer Line during evaluation */}
          {isLoading && (
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 12,
                right: 12,
                height: 2,
                borderRadius: 1,
                backgroundImage: `linear-gradient(90deg, transparent 25%, #70a480 50%, transparent 75%)`,
                backgroundSize: '200% 100%',
                animation: 'halaska-shimmer 1.4s ease-in-out infinite',
              }}
            />
          )}

          {/* Prompt Textarea */}
          <textarea
            ref={activeInputRef}
            rows={1}
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message or address @hush..."
            aria-label="Message input"
            disabled={isLoading}
            style={{
              ...tokens.type.base,
              display: 'block',
              width: '100%',
              boxSizing: 'border-box',
              fontFamily: tokens.font.sans,
              color: pal.text,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              resize: 'none',
              padding: '13px 16px 8px',
              minHeight: 46,
              maxHeight: 140,
              lineHeight: 1.5,
              transition: `color ${motion.smooth} ${motion.easeInOut}`,
            }}
          />

          {/* Halaska Prompt Bottom Action Bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '4px 10px 10px',
            }}
          >
            {/* 1. Speaker Selector Pill */}
            <div ref={speakerRef} style={{ position: 'relative', display: 'inline-flex' }}>
              <button
                type="button"
                onClick={() => setSpeakerMenuOpen((o) => !o)}
                aria-label="Select speaker"
                aria-haspopup="menu"
                aria-expanded={speakerMenuOpen}
                style={{
                  ...interactiveBase,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  height: 28,
                  padding: '0 10px',
                  borderRadius: tokens.radius.pill,
                  background: speakerMenuOpen ? pal.bgMuted : pal.bgSubtle,
                  border: `1px solid ${speakerMenuOpen ? pal.border : pal.borderSubtle}`,
                  color: pal.textSecondary,
                  fontSize: 11,
                  fontFamily: tokens.font.mono,
                }}
              >
                <span>FROM: {sender.toUpperCase()}</span>
                <span style={{ fontSize: 9, opacity: 0.7 }}>▾</span>
              </button>

              {speakerMenuOpen && (
                <div
                  role="menu"
                  style={{
                    position: 'absolute',
                    bottom: 'calc(100% + 6px)',
                    left: 0,
                    zIndex: 30,
                    minWidth: 160,
                    padding: 4,
                    borderRadius: tokens.radius.md,
                    background: '#191919',
                    border: `1px solid ${pal.borderSubtle}`,
                    boxShadow: `0 12px 32px ${pal.shadowLg}`,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                  }}
                >
                  <span
                    style={{
                      padding: '4px 8px',
                      fontSize: 10,
                      fontFamily: tokens.font.mono,
                      letterSpacing: '1px',
                      color: pal.textTertiary,
                    }}
                  >
                    SELECT SENDER
                  </span>
                  {PARTICIPANTS.map((p) => (
                    <button
                      key={p}
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setSender(p);
                        setSpeakerMenuOpen(false);
                      }}
                      style={{
                        ...interactiveBase,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '6px 10px',
                        borderRadius: tokens.radius.sm,
                        textAlign: 'left',
                        background: sender === p ? pal.bgSubtle : 'transparent',
                        color: sender === p ? pal.text : pal.textSecondary,
                        fontSize: 12,
                        fontFamily: tokens.font.sans,
                      }}
                    >
                      <span>{p}</span>
                      {sender === p && <span style={{ color: '#70a480' }}>✓</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* 2. Sample Scenarios Pill */}
            <div ref={scenariosRef} style={{ position: 'relative', display: 'inline-flex' }}>
              <button
                type="button"
                onClick={() => setScenariosMenuOpen((o) => !o)}
                aria-label="Test scenarios"
                aria-haspopup="menu"
                aria-expanded={scenariosMenuOpen}
                style={{
                  ...interactiveBase,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  height: 28,
                  padding: '0 10px',
                  borderRadius: tokens.radius.pill,
                  background: scenariosMenuOpen ? pal.bgMuted : pal.bgSubtle,
                  border: `1px solid ${scenariosMenuOpen ? pal.border : pal.borderSubtle}`,
                  color: pal.textSecondary,
                  fontSize: 11,
                  fontFamily: tokens.font.mono,
                }}
              >
                <span>TEST PROMPTS</span>
                <span style={{ fontSize: 9, opacity: 0.7 }}>▾</span>
              </button>

              {scenariosMenuOpen && (
                <div
                  role="menu"
                  style={{
                    position: 'absolute',
                    bottom: 'calc(100% + 6px)',
                    left: 0,
                    zIndex: 30,
                    width: 320,
                    maxWidth: '90vw',
                    padding: 6,
                    borderRadius: tokens.radius.md,
                    background: '#191919',
                    border: `1px solid ${pal.borderSubtle}`,
                    boxShadow: `0 12px 32px ${pal.shadowLg}`,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 3,
                  }}
                >
                  <span
                    style={{
                      padding: '4px 8px',
                      fontSize: 10,
                      fontFamily: tokens.font.mono,
                      letterSpacing: '1px',
                      color: pal.textTertiary,
                    }}
                  >
                    LOAD SCENARIO
                  </span>
                  {SAMPLE_SCENARIOS.slice(0, 5).map((s) => (
                    <button
                      key={s.id}
                      type="button"
                      role="menuitem"
                      onClick={() => handleSelectScenario(s)}
                      style={{
                        ...interactiveBase,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'flex-start',
                        gap: 2,
                        padding: '6px 8px',
                        borderRadius: tokens.radius.sm,
                        textAlign: 'left',
                        background: 'transparent',
                        color: pal.textSecondary,
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = pal.bgSubtle;
                        e.currentTarget.style.color = pal.text;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent';
                        e.currentTarget.style.color = pal.textSecondary;
                      }}
                    >
                      <span
                        style={{
                          fontSize: 10,
                          fontFamily: tokens.font.mono,
                          color: '#70a480',
                        }}
                      >
                        [{s.sender}]
                      </span>
                      <span
                        style={{
                          fontSize: 11.5,
                          fontFamily: tokens.font.sans,
                          lineHeight: 1.35,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                        }}
                      >
                        {s.text}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Spacer */}
            <div style={{ flex: 1 }} />

            {/* Model Tag */}
            <span
              style={{
                ...tokens.type.xs,
                fontFamily: tokens.font.mono,
                color: pal.textTertiary,
                padding: '2px 8px',
                borderRadius: tokens.radius.pill,
                background: pal.bgSubtle,
                border: `1px solid ${pal.borderSubtle}`,
                userSelect: 'none',
              }}
            >
              llama-3.3-70b
            </span>

            {/* Halaska Circular Send Button */}
            <button
              type="button"
              onClick={() => handleSubmit()}
              disabled={!canSend}
              aria-label={isLoading ? 'Evaluating' : 'Send message'}
              title="Send message (Enter)"
              onMouseEnter={() => setSendHover(true)}
              onMouseLeave={() => setSendHover(false)}
              style={{
                ...interactiveBase,
                width: 32,
                height: 32,
                borderRadius: 16,
                padding: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                background: !canSend
                  ? pal.bgMuted
                  : sendHover
                  ? '#5d8c6b'
                  : '#70a480',
                color: !canSend ? pal.textMuted : '#fff',
                cursor: !canSend ? 'not-allowed' : 'pointer',
                transition: `all ${motion.fast} ${motion.easeInOut}`,
              }}
            >
              {isLoading ? (
                <Spinner size={14} theme={theme} />
              ) : (
                <svg
                  viewBox="0 0 16 16"
                  width="14"
                  height="14"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.75"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="8" y1="12.5" x2="8" y2="3.5" />
                  <polyline points="4.5,7 8,3.5 11.5,7" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
