export interface SampleScenario {
  id: string;
  sender: string;
  text: string;
}

/**
 * Pre-configured sample scenarios for quickly populating the message composer.
 * These act as input helpers only and do not produce mock or canned decisions.
 */
export const SAMPLE_SCENARIOS: SampleScenario[] = [
  {
    id: 's-01',
    sender: 'Alex',
    text: 'The build is broken again.',
  },
  {
    id: 's-02',
    sender: 'Sam',
    text: 'Anyone free for a quick coffee break?',
  },
  {
    id: 's-03',
    sender: 'Taylor',
    text: 'Hush, what is the core responsibility of the decision gate?',
  },
  {
    id: 's-04',
    sender: 'Alex',
    text: 'Thanks Taylor, that helps!',
  },
  {
    id: 's-05',
    sender: 'Jordan',
    text: 'The tests failed in the auth module, status 401 on refresh token.',
  },
  {
    id: 's-06',
    sender: 'You',
    text: 'Can you explain what machine learning is in simple terms?',
  },
];
