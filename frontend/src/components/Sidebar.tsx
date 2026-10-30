import React, { useState } from 'react';
import {
  usePal,
  tokens,
  motion,
  interactiveBase,
  Caption,
  Badge,
  StatusDot,
  IconButton,
  Button,
  Divider,
} from '../halaska-kit';

export type NavTab = 'chat' | 'analytics' | 'how-it-works' | 'prompts' | 'settings';

interface SidebarProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  backendConnected: boolean;
  onReset: () => void;
  generationMode?: string | null;
  onToggleSidebar?: () => void;
  isConversationEmpty?: boolean;
  conversationTitle?: string;
}

const NAV_ITEMS: { id: NavTab; num: string; label: string }[] = [
  { id: 'chat', num: '01', label: 'CHAT' },
  { id: 'analytics', num: '02', label: 'ANALYTICS' },
  { id: 'how-it-works', num: '03', label: 'HOW IT WORKS' },
  { id: 'prompts', num: '04', label: 'PROMPTS' },
  { id: 'settings', num: '05', label: 'SETTINGS' },
];

interface NavItemProps {
  item: { id: NavTab; num: string; label: string };
  isActive: boolean;
  onClick: () => void;
  pal: any;
}

const NavItem: React.FC<NavItemProps> = ({ item, isActive, onClick, pal }) => {
  const [hover, setHover] = useState(false);

  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        ...interactiveBase,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        width: '100%',
        height: 34,
        padding: '0 10px',
        borderRadius: 'var(--radius-sm)',
        background: isActive ? pal.bgMuted : hover ? pal.bgHover : 'transparent',
        border: `1px solid ${isActive ? pal.border : 'transparent'}`,
        textAlign: 'left',
        transition: `all ${motion.smooth} ${motion.easeInOut}`,
      }}
      aria-current={isActive ? 'page' : undefined}
    >
      <span
        style={{
          fontFamily: tokens.font.mono,
          fontSize: 10,
          color: isActive ? pal.text : pal.textTertiary,
          letterSpacing: '0.5px',
          transition: `color ${motion.smooth} ${motion.easeInOut}`,
        }}
      >
        [{item.num}]
      </span>
      <span
        style={{
          flex: 1,
          fontFamily: tokens.font.sans,
          fontSize: 11.5,
          fontWeight: isActive ? 600 : 500,
          letterSpacing: '1px',
          color: isActive || hover ? pal.text : pal.textSecondary,
          transition: `color ${motion.smooth} ${motion.easeInOut}`,
        }}
      >
        {item.label}
      </span>
      <span
        style={{
          fontSize: 11,
          color: isActive ? pal.text : pal.textTertiary,
          transform: hover ? 'translate(1px, -1px)' : 'none',
          transition: `all ${motion.smooth} ${motion.easeInOut}`,
        }}
        aria-hidden="true"
      >
        ↗
      </span>
    </button>
  );
};

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  backendConnected,
  onReset,
  generationMode,
  onToggleSidebar,
  isConversationEmpty = true,
  conversationTitle,
}) => {
  const theme = 'dark';
  const pal = usePal(theme);

  const showConversationBadge = isConversationEmpty || !!conversationTitle;
  const badgeLabel = isConversationEmpty ? 'NEW CONVERSATION' : conversationTitle;

  return (
    <aside
      className="sidebar-container"
      aria-label="Sidebar navigation and status"
      style={{
        background: pal.bgSubtle,
        borderRight: `1px solid ${pal.borderSubtle}`,
        fontFamily: tokens.font.sans,
        color: pal.text,
      }}
    >
      {/* 1. Brand Header: Logo, Wordmark, and Hamburger Toggle */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
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
          {onToggleSidebar && (
            <IconButton
              icon="☰"
              size={28}
              label="Hide sidebar"
              theme={theme}
              onClick={onToggleSidebar}
              style={{ color: pal.textSecondary }}
            />
          )}
        </div>

        {/* 2. Short Tagline */}
        <div
          style={{
            fontFamily: tokens.font.mono,
            fontSize: 10,
            letterSpacing: '1.5px',
            color: pal.textTertiary,
            textTransform: 'uppercase',
            marginTop: 8,
          }}
        >
          DECIDES BEFORE IT SPEAKS.
        </div>

        {/* 3. Connection and Conversation Status */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 8,
            marginTop: 20,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <StatusDot
              status={backendConnected ? 'online' : 'offline'}
              pulse={backendConnected}
              size={6}
              theme={theme}
            />
            <span
              style={{
                fontFamily: tokens.font.mono,
                fontSize: 10.5,
                letterSpacing: '0.8px',
                color: backendConnected ? '#70a480' : pal.textTertiary,
                whiteSpace: 'nowrap',
              }}
            >
              {backendConnected ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          {showConversationBadge && (
            <Badge
              theme={theme}
              style={{
                fontFamily: tokens.font.mono,
                fontSize: 9.5,
                letterSpacing: '0.6px',
                padding: '2px 7px',
                borderRadius: 'var(--radius-sm)',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${pal.borderSubtle}`,
                color: pal.textTertiary,
                textTransform: 'uppercase',
                whiteSpace: 'nowrap',
                fontWeight: tokens.weight.medium,
              }}
            >
              {badgeLabel}
            </Badge>
          )}
        </div>

        <Divider theme={theme} spacing={12} />
      </div>

      {/* 2. Navigation Links */}
      <nav
        aria-label="Main Navigation"
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
        }}
      >
        {NAV_ITEMS.map((item) => (
          <NavItem
            key={item.id}
            item={item}
            isActive={activeTab === item.id}
            onClick={() => onSelectTab(item.id)}
            pal={pal}
          />
        ))}
      </nav>

      {/* 3. Footer: Reset Session */}
      <div
        style={{
          marginTop: 'auto',
          paddingTop: 16,
          borderTop: `1px solid ${pal.borderSubtle}`,
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
      >
        {generationMode === 'stub' && (
          <Caption
            theme={theme}
            style={{
              ...tokens.type.xs,
              fontFamily: tokens.font.mono,
              color: '#be9c56',
            }}
            role="status"
          >
            [Demo mode active]
          </Caption>
        )}
        <Button
          variant="secondary"
          size="sm"
          fullWidth
          theme={theme}
          onClick={onReset}
          style={{
            fontFamily: tokens.font.mono,
            fontSize: 10,
            letterSpacing: '1px',
            color: pal.textSecondary,
          }}
          aria-label="Reset session"
        >
          RESET SESSION
        </Button>
      </div>
    </aside>
  );
};
