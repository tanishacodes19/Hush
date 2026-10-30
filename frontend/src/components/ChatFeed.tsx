import React, { useEffect, useRef } from 'react';
import { FeedMessage } from '../types/api';
import {
  usePal,
  tokens,
  motion,
  Text,
  Badge,
  StatusDot,
  IconButton,
  AlertBanner,
  ThinkingIndicator,
} from '../halaska-kit';

export interface SessionStatsData {
  messagesEvaluated: number;
  speak: number;
  ask: number;
  silent: number;
  generationCalls: number;
  callsSaved: number;
  reductionPercentage: number;
}

interface ChatFeedProps {
  messages: FeedMessage[];
  isLoading?: boolean;
  onToggleSidebar?: () => void;
  sidebarOpen?: boolean;
  backendConnected?: boolean;
  onReset?: () => void;
  currentDecision?: 'SPEAK' | 'ASK' | 'SILENT' | null;
  sessionStats?: SessionStatsData;
}

const DECISION_BADGES = [
  {
    id: 'SPEAK' as const,
    label: 'SPEAK',
    color: '#70a480',
    bg: 'rgba(112, 164, 128, 0.12)',
    border: 'rgba(112, 164, 128, 0.3)',
  },
  {
    id: 'ASK' as const,
    label: 'ASK',
    color: '#be9c56',
    bg: 'rgba(190, 156, 86, 0.12)',
    border: 'rgba(190, 156, 86, 0.3)',
  },
  {
    id: 'SILENT' as const,
    label: 'SILENT',
    color: '#8e82b0',
    bg: 'rgba(142, 130, 176, 0.12)',
    border: 'rgba(142, 130, 176, 0.3)',
  },
];

const formatLatency = (ms: number | null | undefined): string => {
  if (ms === null || ms === undefined) return '—';
  if (ms < 1) return '<1 MS';
  return `${Math.round(ms)} MS`;
};

const formatConfidence = (conf: number | null | undefined): string => {
  if (conf === null || conf === undefined) return '—';
  return conf.toFixed(2);
};

export const ChatFeed: React.FC<ChatFeedProps> = ({
  messages,
  isLoading = false,
  onToggleSidebar,
  sidebarOpen = true,
  currentDecision,
  sessionStats,
}) => {
  const theme = 'dark';
  const pal = usePal(theme);
  const bottomRef = useRef<HTMLDivElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const stats = sessionStats ?? {
    messagesEvaluated: 0,
    speak: 0,
    ask: 0,
    silent: 0,
    generationCalls: 0,
    callsSaved: 0,
    reductionPercentage: 0,
  };

  const activeBadge = DECISION_BADGES.find((d) => d.id === currentDecision);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        height: '100%',
        width: '100%',
        background: pal.bg,
        fontFamily: tokens.font.sans,
        color: pal.text,
        overflow: 'hidden',
      }}
    >
      {/* ── Main Chat Header: Decision and Session Statistics ───────── */}
      <header
        style={{
          minHeight: 46,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 16,
          padding: '8px 28px',
          borderBottom: `1px solid ${pal.borderSubtle}`,
          background: pal.bgSubtle,
          transition: `all ${motion.smooth} ${motion.easeInOut}`,
          zIndex: 10,
        }}
        role="region"
        aria-label="Decision and session statistics"
      >
        {/* Left Group: Current Decision */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {/* Collapsed Sidebar Branding: 1. Hamburger -> 2. Logo -> 3. Wordmark */}
          <div
            className={`collapsed-header-brand ${!sidebarOpen ? 'is-visible' : 'is-hidden'}`}
            aria-hidden={sidebarOpen ? 'true' : 'false'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              maxWidth: !sidebarOpen ? 220 : 0,
              marginRight: !sidebarOpen ? 6 : 0,
              opacity: !sidebarOpen ? 1 : 0,
              pointerEvents: !sidebarOpen ? 'auto' : 'none',
              overflow: 'hidden',
              whiteSpace: 'nowrap',
              transition: !sidebarOpen
                ? 'opacity 250ms ease-in 100ms, max-width var(--sidebar-transition-duration, 300ms) var(--sidebar-transition-timing, cubic-bezier(0.22, 1, 0.36, 1)), margin-right var(--sidebar-transition-duration, 300ms) var(--sidebar-transition-timing, cubic-bezier(0.22, 1, 0.36, 1))'
                : 'opacity 150ms ease-out, max-width var(--sidebar-transition-duration, 300ms) var(--sidebar-transition-timing, cubic-bezier(0.22, 1, 0.36, 1)), margin-right var(--sidebar-transition-duration, 300ms) var(--sidebar-transition-timing, cubic-bezier(0.22, 1, 0.36, 1))',
            }}
          >
            {onToggleSidebar && (
              <IconButton
                icon="☰"
                size={28}
                label="Show sidebar"
                theme={theme}
                onClick={onToggleSidebar}
                style={{ color: pal.textSecondary, flexShrink: 0 }}
              />
            )}

            <div
              onClick={onToggleSidebar}
              title={onToggleSidebar ? 'Show sidebar' : undefined}
              role={onToggleSidebar ? 'button' : undefined}
              tabIndex={onToggleSidebar && !sidebarOpen ? 0 : -1}
              onKeyDown={
                onToggleSidebar
                  ? (e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onToggleSidebar();
                      }
                    }
                  : undefined
              }
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                userSelect: 'none',
                cursor: onToggleSidebar ? 'pointer' : 'default',
                flexShrink: 0,
              }}
            >
              <img
                src="/favicon.svg"
                alt="Hush logo"
                width={22}
                height={22}
                style={{ display: 'block', borderRadius: 4, flexShrink: 0 }}
              />
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 2 }}>
                <span
                  style={{
                    fontFamily: tokens.font.sans,
                    fontSize: 15,
                    fontWeight: 600,
                    letterSpacing: '2.2px',
                    color: pal.text,
                  }}
                >
                  HUSH
                </span>
                <span
                  style={{
                    fontFamily: tokens.font.sans,
                    fontSize: 10,
                    color: pal.textTertiary,
                  }}
                >
                  ®
                </span>
              </div>
            </div>
          </div>

          <span
            style={{
              fontFamily: tokens.font.sans,
              fontSize: 11.5,
              fontWeight: 500,
              color: pal.textTertiary,
              letterSpacing: '0.3px',
            }}
          >
            CURRENT DECISION:
          </span>

          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            {DECISION_BADGES.map((d) => {
              const isActive = currentDecision === d.id;
              return (
                <span
                  key={d.id}
                  style={{
                    fontFamily: tokens.font.mono,
                    fontSize: 10.5,
                    letterSpacing: '0.6px',
                    padding: '3px 8px',
                    borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${isActive ? d.border : pal.borderSubtle}`,
                    background: isActive ? d.bg : 'transparent',
                    color: isActive ? d.color : pal.textSecondary,
                    fontWeight: isActive ? 600 : 400,
                    transition: `all ${motion.smooth} ${motion.easeInOut}`,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 5,
                  }}
                  title={isActive ? `Active decision: ${d.label}` : d.label}
                >
                  <span
                    style={{
                      width: 5,
                      height: 5,
                      borderRadius: 1,
                      background: isActive ? d.color : pal.textTertiary,
                      opacity: isActive ? 1 : 0.4,
                      boxShadow: isActive ? `0 0 6px ${d.color}60` : 'none',
                    }}
                    aria-hidden="true"
                  />
                  {d.label}
                </span>
              );
            })}
          </div>

          <span
            style={{
              fontFamily: tokens.font.sans,
              fontSize: 11,
              color: currentDecision ? activeBadge?.color : pal.textTertiary,
              fontWeight: currentDecision ? 500 : 400,
            }}
          >
            {currentDecision ? `· Active: ${currentDecision}` : '· NO DECISION YET'}
          </span>
        </div>

        {/* Right Group: Session Statistics */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            flexWrap: 'wrap',
            fontSize: 11,
          }}
        >
          {/* Messages Evaluated + Compact Secondary Decisions */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: pal.textTertiary, fontFamily: tokens.font.sans, letterSpacing: '0.2px' }}>
              MESSAGES EVALUATED:
            </span>
            <strong style={{ fontFamily: tokens.font.mono, color: pal.text, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
              {stats.messagesEvaluated}
            </strong>
            <span
              style={{
                fontFamily: tokens.font.mono,
                fontSize: 10.5,
                color: pal.textTertiary,
              }}
              title="Decision counts: Speak / Ask / Silent"
            >
              (<span style={{ color: '#70a480' }}>{stats.speak}</span> speak · <span style={{ color: '#be9c56' }}>{stats.ask}</span> ask · <span style={{ color: '#8e82b0' }}>{stats.silent}</span> silent)
            </span>
          </div>

          <span style={{ color: pal.borderSubtle }} aria-hidden="true">·</span>

          {/* Generation Calls */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: pal.textTertiary, fontFamily: tokens.font.sans, letterSpacing: '0.2px' }}>
              GENERATION CALLS:
            </span>
            <strong style={{ fontFamily: tokens.font.mono, color: pal.text, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
              {stats.generationCalls}
            </strong>
          </div>

          <span style={{ color: pal.borderSubtle }} aria-hidden="true">·</span>

          {/* Calls Saved */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: pal.textTertiary, fontFamily: tokens.font.sans, letterSpacing: '0.2px' }}>
              CALLS SAVED:
            </span>
            <strong style={{ fontFamily: tokens.font.mono, color: '#70a480', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
              {stats.callsSaved}
            </strong>
          </div>

          <span style={{ color: pal.borderSubtle }} aria-hidden="true">·</span>

          {/* Reduction Percentage */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: pal.textTertiary, fontFamily: tokens.font.sans, letterSpacing: '0.2px' }}>
              REDUCTION PERCENTAGE:
            </span>
            <strong
              style={{
                fontFamily: tokens.font.mono,
                fontSize: 11,
                color: stats.reductionPercentage > 0 ? '#70a480' : pal.text,
                fontWeight: 600,
                fontVariantNumeric: 'tabular-nums',
                background: stats.reductionPercentage > 0 ? 'rgba(112, 164, 128, 0.12)' : 'transparent',
                padding: '2px 7px',
                borderRadius: 'var(--radius-sm)',
                border: `1px solid ${stats.reductionPercentage > 0 ? 'rgba(112, 164, 128, 0.28)' : pal.borderSubtle}`,
              }}
            >
              {stats.reductionPercentage}%
            </strong>
          </div>
        </div>
      </header>

      {/* ── Halaska Conversation Thread Area (Expanded) ───────────── */}
      <div
        ref={threadRef}
        className="chat-scroll-area"
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          padding: '28px 28px 20px',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          style={{
            maxWidth: 760,
            width: '100%',
            margin: '0 auto',
            display: 'flex',
            flexDirection: 'column',
            flex: messages.length === 0 && !isLoading ? 1 : undefined,
            justifyContent: messages.length === 0 && !isLoading ? 'center' : undefined,
            gap: 22,
          }}
          role="log"
          aria-live="polite"
        >
          {/* Empty Conversation Minimal Brand Identity */}
          {messages.length === 0 && !isLoading && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '48px 16px',
                textAlign: 'center',
                gap: 12,
                userSelect: 'none',
                animation: `halaska-step-in 0.4s ${motion.emphasized} both`,
              }}
            >
              {/* 1. Official Hush Logo */}
              <img
                src="/favicon.svg"
                alt="Hush logo"
                width={52}
                height={52}
                style={{
                  display: 'block',
                  borderRadius: 10,
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
                }}
              />

              {/* 2. HUSH Wordmark */}
              <div
                style={{
                  fontFamily: tokens.font.sans,
                  fontSize: 15,
                  fontWeight: 600,
                  letterSpacing: '2.5px',
                  color: pal.text,
                  marginTop: 4,
                }}
              >
                HUSH
              </div>

              {/* 3. Tagline */}
              <div
                style={{
                  fontFamily: tokens.font.mono,
                  fontSize: 11,
                  fontWeight: 500,
                  letterSpacing: '1.8px',
                  color: pal.textSecondary,
                }}
              >
                DECIDES BEFORE IT SPEAKS.
              </div>

              {/* 4. Supporting Line */}
              <div
                style={{
                  fontFamily: tokens.font.sans,
                  fontSize: 12,
                  letterSpacing: '0.3px',
                  color: pal.textTertiary,
                  marginTop: 2,
                }}
              >
                A little more thought. A little less noise.
              </div>
            </div>
          )}

          {/* Messages Feed */}
          {messages.map((msg) => {
            // System Error Row
            if (msg.isError) {
              return (
                <div
                  key={msg.id}
                  style={{
                    width: '100%',
                    maxWidth: '82%',
                    animation: `halaska-step-in 0.35s ${motion.emphasized} both`,
                  }}
                  role="alert"
                >
                  <AlertBanner
                    variant="danger"
                    title="System Notification"
                    description={msg.text}
                    theme={theme}
                  />
                </div>
              );
            }

            const isHush = msg.isAgent;
            const action = msg.decision?.action;
            const isSilent = isHush && action === 'SILENT';
            const confidence = msg.decision ? formatConfidence(msg.decision.confidence) : null;
            const latency = msg.decision ? formatLatency(msg.decision.decision_latency_ms) : null;

            // 1. User Turn: Right-aligned bubble matching Halaska ChatParadigmExample
            if (!isHush) {
              return (
                <article
                  key={msg.id}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-end',
                    gap: 4,
                    animation: `halaska-step-in 0.35s ${motion.emphasized} both`,
                  }}
                >
                  <div
                    style={{
                      maxWidth: '78%',
                      padding: '12px 16px',
                      background: '#181818',
                      border: '1px solid #30302d',
                      borderRadius: 'var(--radius-sm)',
                      transition: `all ${motion.smooth} ${motion.easeInOut}`,
                    }}
                  >
                    <Text size="base" theme={theme} style={{ color: pal.text, lineHeight: 1.55 }}>
                      {msg.text}
                    </Text>
                  </div>
                  <span
                    style={{
                      ...tokens.type.xs,
                      fontFamily: tokens.font.mono,
                      color: pal.textTertiary,
                      paddingRight: 4,
                      transition: `color ${motion.smooth} ${motion.easeInOut}`,
                    }}
                  >
                    {msg.sender && msg.sender !== 'You' ? `${msg.sender} · ` : ''}
                    {msg.timestamp}
                  </span>
                </article>
              );
            }

            // 2. Hush Agent Turn: Left-aligned agent card matching Halaska ChatParadigmExample
            const getActionStyle = () => {
              if (action === 'SPEAK') {
                return {
                  badgeColor: '#70a480',
                  badgeBg: 'rgba(112, 164, 128, 0.12)',
                  badgeBorder: 'rgba(112, 164, 128, 0.3)',
                  statusDot: 'online' as const,
                };
              }
              if (action === 'ASK') {
                return {
                  badgeColor: '#be9c56',
                  badgeBg: 'rgba(190, 156, 86, 0.12)',
                  badgeBorder: 'rgba(190, 156, 86, 0.3)',
                  statusDot: 'busy' as const,
                };
              }
              return {
                badgeColor: '#8e82b0',
                badgeBg: 'rgba(142, 130, 176, 0.12)',
                badgeBorder: 'rgba(142, 130, 176, 0.3)',
                statusDot: 'offline' as const,
              };
            };

            const actionStyle = getActionStyle();

            return (
              <article
                key={msg.id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  gap: 8,
                  width: '100%',
                  animation: `halaska-step-in 0.35s ${motion.emphasized} both`,
                }}
              >
                {/* Agent Header */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingLeft: 2 }}>
                  <img
                    src="/favicon.svg"
                    alt="Hush logo"
                    width={20}
                    height={20}
                    style={{
                      display: 'block',
                      borderRadius: 4,
                      flexShrink: 0,
                    }}
                  />
                  <Text size="sm" weight="semibold" theme={theme}>
                    HUSH
                  </Text>
                  {action && (
                    <Badge
                      theme={theme}
                      style={{
                        color: actionStyle.badgeColor,
                        background: actionStyle.badgeBg,
                        border: `1px solid ${actionStyle.badgeBorder}`,
                      }}
                    >
                      <StatusDot status={actionStyle.statusDot} size={6} theme={theme} />
                      {action}
                    </Badge>
                  )}
                  <span
                    style={{
                      ...tokens.type.xs,
                      fontFamily: tokens.font.mono,
                      color: pal.textTertiary,
                    }}
                  >
                    {msg.timestamp}
                  </span>
                </div>

                {/* Agent Message Card */}
                <div
                  style={{
                    width: '100%',
                    maxWidth: '82%',
                    background: isSilent ? '#141414' : '#161616',
                    border: `1px solid ${isSilent ? 'rgba(142, 130, 176, 0.28)' : '#30302d'}`,
                    borderRadius: 'var(--radius-sm)',
                    padding: '14px 18px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 10,
                    transition: `all ${motion.smooth} ${motion.easeInOut}`,
                  }}
                >
                  {isSilent ? (
                    <div
                      style={{
                        fontStyle: 'italic',
                        color: pal.textSecondary,
                        fontSize: 13,
                        lineHeight: 1.55,
                      }}
                    >
                      (Hush evaluated the context and chose to remain silent. No generation call was made.)
                    </div>
                  ) : (
                    <div
                      style={{
                        fontSize: 13.5,
                        lineHeight: 1.6,
                        color: pal.text,
                        wordBreak: 'break-word',
                      }}
                    >
                      {msg.text.split('\n').map((line, idx) => (
                        <p key={idx} style={{ margin: idx > 0 ? '6px 0 0' : 0 }}>
                          {line}
                        </p>
                      ))}
                    </div>
                  )}

                  {/* Telemetry Row */}
                  {msg.decision && (
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 10,
                        paddingTop: 8,
                        borderTop: `1px solid ${pal.borderSubtle}`,
                        fontSize: 10.5,
                        fontFamily: tokens.font.mono,
                        color: pal.textTertiary,
                      }}
                    >
                      <span>CONFIDENCE {confidence}</span>
                      <span>•</span>
                      <span>LATENCY {latency}</span>
                      <span>•</span>
                      <span>{msg.decision.generation_used ? 'GENERATED' : 'CALL SAVED'}</span>
                    </div>
                  )}
                </div>
              </article>
            );
          })}

          {/* Halaska Thinking / Evaluating Turn */}
          {isLoading && (
            <article
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                gap: 8,
                width: '100%',
                animation: `halaska-step-in 0.35s ${motion.emphasized} both`,
              }}
              role="status"
              aria-label="Hush is evaluating"
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingLeft: 2 }}>
                <img
                  src="/favicon.svg"
                  alt="Hush logo"
                  width={20}
                  height={20}
                  style={{
                    display: 'block',
                    borderRadius: 4,
                    flexShrink: 0,
                  }}
                />
                <Text size="sm" weight="semibold" theme={theme}>
                  HUSH
                </Text>
                <Badge
                  theme={theme}
                  style={{
                    color: '#70a480',
                    background: 'rgba(112, 164, 128, 0.12)',
                    border: '1px solid rgba(112, 164, 128, 0.28)',
                  }}
                >
                  <StatusDot status="online" pulse size={6} theme={theme} />
                  EVALUATING
                </Badge>
              </div>

              <div
                style={{
                  width: '100%',
                  maxWidth: '82%',
                  background: '#151515',
                  border: `1px solid ${pal.borderSubtle}`,
                  borderRadius: 'var(--radius-sm)',
                  padding: '14px 18px',
                }}
              >
                <ThinkingIndicator
                  label="Evaluating conversational context and turn dynamics…"
                  theme={theme}
                />
              </div>
            </article>
          )}

          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
};
