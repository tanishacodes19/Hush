import { ChatMessageRequest, ChatMessageResponse } from '../types/api';

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL =
  rawBaseUrl !== undefined && rawBaseUrl.trim() !== ''
    ? rawBaseUrl.trim().replace(/\/+$/, '')
    : (import.meta.env.DEV ? 'http://localhost:8000' : '');

export interface SendMessageResult {
  response: ChatMessageResponse;
  isLive: boolean;
}

export interface HealthResponse {
  status: string;
  generation_mode?: string;
  decider_mode?: string;
}

/**
 * Fetch health status and active operating modes from the backend.
 * Safely returns null on network failure or if the endpoint is unreachable.
 */
export async function checkHealth(): Promise<HealthResponse | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Safely return null on failure
  }
  return null;
}

export type ApiFailureCategory =
  | 'unreachable'
  | 'timeout'
  | 'rate_limited'
  | 'client_error'
  | 'server_error'
  | 'config_error';

export class ApiError extends Error {
  readonly category: ApiFailureCategory;
  readonly status?: number;

  constructor(category: ApiFailureCategory, message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.category = category;
    this.status = status;
  }
}

/**
 * Fetch telemetry metrics from the backend.
 * Safely returns null on network failure or if the endpoint is unreachable.
 */
export async function getMetrics(): Promise<any | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/metrics`);
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Safely return null on failure
  }
  return null;
}

function getRequestTimeoutMs(): number {
  const raw = import.meta.env.VITE_REQUEST_TIMEOUT_MS;
  if (raw) {
    const parsed = parseInt(raw, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  // Default timeout: 75000ms (75 seconds).
  // Must exceed the backend OPENAI_TIMEOUT_SECONDS (typically 15-60s) to allow complete
  // response generation from local LLMs on CPU without premature client aborts.
  return 75000;
}

/**
 * Send message to FastAPI backend.
 * Never falls back to local simulation on failure.
 * On failure, throws a typed ApiError with category and optional status.
 */
export async function sendMessage(request: ChatMessageRequest): Promise<ChatMessageResponse> {
  const controller = new AbortController();
  // Request abort timeout must exceed backend OPENAI_TIMEOUT_SECONDS
  const timeoutMs = getRequestTimeoutMs();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (request.chat_context) {
      headers['X-Hush-Chat-Context'] = request.chat_context;
    }

    const { chat_context, ...bodyPayload } = request;

    res = await fetch(`${API_BASE_URL}/api/chat/message`, {
      method: 'POST',
      headers,
      body: JSON.stringify(bodyPayload),
      signal: controller.signal,
    });
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError('timeout', 'Request timed out waiting for backend response.');
    }
    throw new ApiError('unreachable', 'Hush backend is unreachable.');
  } finally {
    clearTimeout(timeoutId);
  }

  if (!res.ok) {
    if (res.status === 429) {
      throw new ApiError('rate_limited', 'Rate limit exceeded.', 429);
    }
    if (res.status >= 500) {
      let detail = '';
      try {
        const body = await res.json();
        detail = body.detail || '';
      } catch {
        // ignore json parse failure
      }
      if (detail.toLowerCase().includes('configuration')) {
        throw new ApiError('config_error', detail, res.status);
      }
      throw new ApiError('server_error', `Backend server error (${res.status}).`, res.status);
    }
    throw new ApiError('client_error', `Backend request rejected (${res.status}).`, res.status);
  }

  return await res.json();
}
