# Hush Frontend

React + Vite frontend for **Hush — The AI That Earns Its Right to Speak**.

Developed by the **Backend & Frontend Developer**.

## Features

- **Live Conversation Feed**: Displays participant dialogue with avatar identifiers.
- **Decision Badges & Inline Restraint Tag**: Visualizes when Hush chooses `SPEAK`, `ASK`, or remains `SILENT` (with generation strictly bypassed).
- **Live Confidence Meter**: Prominently displays calibrated decision confidence (or `N/A` when unavailable) with a circular gauge and latency telemetry.
- **Restraint Metrics Dashboard**: Tracks messages evaluated, action breakdown, generation calls avoided, and call reduction rate.
- **Side-by-Side Comparison View**: Demonstrates the difference between an unconstrained "Always-Reply" bot vs. Hush's calibrated restraint policy.
- **Offline UI Preview Mode**: When the backend is offline or during standalone testing, provides local pre-calibrated sample scenarios without requiring AWS credentials or live model endpoints. All preview metrics are explicitly labeled.

## Prerequisites

- Node.js (v18+)
- npm (v9+)

## Setup & Running

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the local development server:**
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

3. **Build for production:**
   ```bash
   npm run build
   ```

4. **Preview production build:**
   ```bash
   npm run preview
   ```

## Architecture & Integration

All API calls are routed through `src/services/api.ts` to `http://localhost:8000/api/chat/message`. Types in `src/types/api.ts` strictly match `PLAN.md` schemas.
