/**
 * TypeScript API contracts strictly matching PLAN.md Sections 1, 5, and 7.
 */

export type Action = 'SPEAK' | 'SILENT' | 'ASK';

export interface ChatMessageRequest {
  conversation_id: string;
  user_id: string;
  message: string;
}

export interface ChatMessageResponse {
  conversation_id: string;
  action: Action;
  confidence: number | null;
  confidence_kind?: string;
  reply: string | null;
  generation_used: boolean;
  fallback_used: boolean;
  decision_latency_ms: number | null;
}

export interface FeedMessage {
  id: string;
  sender: string;
  text: string;
  timestamp: string;
  isAgent?: boolean;
  isError?: boolean;
  decision?: ChatMessageResponse;
}

export interface RestraintMetricsData {
  messagesEvaluated: number;
  decisions: {
    SPEAK: number;
    SILENT: number;
    ASK: number;
  };
  generationCallsMade: number;
  generationCallsSaved: number;
  callReductionRate: number; // percentage (0 - 100)
  isLive: boolean;
}
