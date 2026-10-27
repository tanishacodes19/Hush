import { ChatMessageRequest, ChatMessageResponse } from '../types/api';
import { SAMPLE_SCENARIOS } from '../data/sampleScenarios';

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL =
  rawBaseUrl !== undefined && rawBaseUrl.trim() !== ''
    ? rawBaseUrl.trim().replace(/\/+$/, '')
    : (import.meta.env.DEV ? 'http://localhost:8000' : '');

export interface SendMessageResult {
  response: ChatMessageResponse;
  isLive: boolean;
}

/**
 * Send message to FastAPI backend.
 * Falls back to UI Preview Mode with clear labeling if backend is offline.
 */
export async function sendMessage(request: ChatMessageRequest): Promise<SendMessageResult> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    const res = await fetch(`${API_BASE_URL}/api/chat/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (res.ok) {
      const data: ChatMessageResponse = await res.json();
      return { response: data, isLive: true };
    }
  } catch {
    // Backend unreachable: fall back to local preview simulation
  }

  // UI Preview Mode Fallback:
  // Match a pre-configured sample scenario, or compute a predictable offline preview decision.
  // NOTE: In UI Preview Mode, benchmark fields like decision_latency_ms are set to null.
  const matched = SAMPLE_SCENARIOS.find(
    (s) => s.text.toLowerCase().trim() === request.message.toLowerCase().trim()
  );

  if (matched) {
    return {
      response: {
        ...matched.mockDecision,
        conversation_id: request.conversation_id,
        decision_latency_ms: null, // Latency not measured in preview mode
      },
      isLive: false,
    };
  }

  // Heuristic mock decision for custom typed text in Preview Mode:
  const isDirect = request.message.toLowerCase().includes('hush') || request.message.startsWith('@hush');
  const hasQuestion = request.message.includes('?');

  if (isDirect) {
    return {
      response: {
        conversation_id: request.conversation_id,
        action: 'SPEAK',
        confidence: 0.91, // Illustrative preview value
        reply: `[UI Preview] Received direct request: "${request.message}". Bedrock would generate the response here.`,
        generation_used: true,
        fallback_used: false,
        decision_latency_ms: null, // Not measured in preview mode
      },
      isLive: false,
    };
  } else if (hasQuestion) {
    return {
      response: {
        conversation_id: request.conversation_id,
        action: 'ASK',
        confidence: 0.81, // Illustrative preview value
        reply: `[UI Preview] Could you provide more context on "${request.message}"?`,
        generation_used: true,
        fallback_used: false,
        decision_latency_ms: null, // Not measured in preview mode
      },
      isLive: false,
    };
  } else {
    return {
      response: {
        conversation_id: request.conversation_id,
        action: 'SILENT',
        confidence: 0.94, // Illustrative preview value
        reply: null,
        generation_used: false,
        fallback_used: false,
        decision_latency_ms: null, // Not measured in preview mode
      },
      isLive: false,
    };
  }
}
