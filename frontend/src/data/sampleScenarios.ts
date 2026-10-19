import { ChatMessageResponse } from '../types/api';

export interface SampleScenario {
  id: string;
  sender: string;
  text: string;
  mockDecision: ChatMessageResponse;
  alwaysReplyText: string;
}

/**
 * Pre-configured sample scenarios for offline UI Preview Mode.
 *
 * NOTE: These are local UI preview samples only.
 * Confidence scores are illustrative sample values.
 * Decision latency is set to null because benchmarks are not measured in preview mode.
 */
export const SAMPLE_SCENARIOS: SampleScenario[] = [
  {
    id: 's-01',
    sender: 'Alex',
    text: 'The build is broken again.',
    mockDecision: {
      conversation_id: 'demo-room',
      action: 'ASK',
      confidence: 0.84, // Illustrative preview value
      reply: 'Do you have the deployment log or the specific error message?',
      generation_used: true,
      fallback_used: false,
      decision_latency_ms: null, // Not measured in preview mode
    },
    alwaysReplyText: 'I am an AI assistant. I noticed you mentioned the build is broken. How can I help you fix it?',
  },
  {
    id: 's-02',
    sender: 'Sam',
    text: 'Anyone free for a quick coffee break?',
    mockDecision: {
      conversation_id: 'demo-room',
      action: 'SILENT',
      confidence: 0.96, // Illustrative preview value
      reply: null,
      generation_used: false,
      fallback_used: false,
      decision_latency_ms: null, // Not measured in preview mode
    },
    alwaysReplyText: 'I would love to join for coffee, but as an AI model I cannot drink coffee or take breaks!',
  },
  {
    id: 's-03',
    sender: 'Taylor',
    text: 'Hush, what is the core responsibility of the decision gate?',
    mockDecision: {
      conversation_id: 'demo-room',
      action: 'SPEAK',
      confidence: 0.95, // Illustrative preview value
      reply: 'The decision gate evaluates conversation state to decide whether Hush should SPEAK, ASK, or stay SILENT before generating text.',
      generation_used: true,
      fallback_used: false,
      decision_latency_ms: null, // Not measured in preview mode
    },
    alwaysReplyText: 'The decision gate evaluates conversation state.',
  },
  {
    id: 's-04',
    sender: 'Alex',
    text: 'Thanks Taylor, that helps!',
    mockDecision: {
      conversation_id: 'demo-room',
      action: 'SILENT',
      confidence: 0.93, // Illustrative preview value
      reply: null,
      generation_used: false,
      fallback_used: false,
      decision_latency_ms: null, // Not measured in preview mode
    },
    alwaysReplyText: "You're very welcome! Let me know if you need anything else.",
  },
  {
    id: 's-05',
    sender: 'Jordan',
    text: 'The tests failed in the auth module, status 401 on refresh token.',
    mockDecision: {
      conversation_id: 'demo-room',
      action: 'SPEAK',
      confidence: 0.89, // Illustrative preview value
      reply: 'A 401 on token refresh typically indicates an expired secret key or mismatched audience claim in the JWT validator.',
      generation_used: true,
      fallback_used: false,
      decision_latency_ms: null, // Not measured in preview mode
    },
    alwaysReplyText: 'Error 401 Unauthorized means authentication failed. Check your credentials.',
  },
];
