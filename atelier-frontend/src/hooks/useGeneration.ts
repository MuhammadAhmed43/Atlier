import { useCallback, useEffect, useRef, useState } from 'react';
import { buildStreamUrl } from '../api';
import type {
  DoneEvent,
  ErrorEvent as ServerErrorEvent,
  GenerationParams,
  GenerationPhase,
  ReadyEvent,
  StepEvent,
} from '../types';

export type PreviewFrame = { index: number; dataUrl: string };

export type GenerationState = {
  phase: GenerationPhase;
  /** total denoising steps for the active run */
  total: number;
  /** latest step index (1-based) reported by the backend */
  step: number;
  /** wall-clock elapsed seconds since stream opened */
  elapsedS: number;
  /** PNG data URLs from the `ready` event */
  reference: string | null;
  edges: string | null;
  /** model-side dimensions of the active generation */
  width: number;
  height: number;
  /** width / height ratio — used to size the theatre canvas dynamically */
  aspect: number;
  /** final image (from `done` event) */
  finalImage: string | null;
  finalSeed: number | null;
  /** preview frames accumulated from `step` events that carried preview_b64 */
  previews: PreviewFrame[];
  /** the most-recent preview frame (used for the big canvas) */
  latestPreview: string | null;
  error: string | null;
};

const initialState: GenerationState = {
  phase: 'idle',
  total: 0,
  step: 0,
  elapsedS: 0,
  reference: null,
  edges: null,
  width: 0,
  height: 0,
  aspect: 1,
  finalImage: null,
  finalSeed: null,
  previews: [],
  latestPreview: null,
  error: null,
};

export function useGeneration() {
  const [state, setState] = useState<GenerationState>(initialState);
  const esRef = useRef<EventSource | null>(null);
  const startedAtRef = useRef<number>(0);
  const tickRef = useRef<number | null>(null);
  /** Flips true once we've received a `done` event; suppresses post-close error noise. */
  const doneRef = useRef<boolean>(false);

  const stop = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (tickRef.current != null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    stop();
    setState(initialState);
  }, [stop]);

  const start = useCallback(
    (params: GenerationParams) => {
      stop();
      doneRef.current = false;
      setState({ ...initialState, phase: 'preparing' });
      startedAtRef.current = performance.now();

      tickRef.current = window.setInterval(() => {
        setState((s) =>
          s.phase === 'generating' || s.phase === 'deconstructing' || s.phase === 'preparing'
            ? { ...s, elapsedS: (performance.now() - startedAtRef.current) / 1000 }
            : s,
        );
      }, 100);

      const es = new EventSource(buildStreamUrl(params));
      esRef.current = es;

      es.addEventListener('ready', (e) => {
        const d = JSON.parse((e as MessageEvent).data) as ReadyEvent;
        setState((s) => ({
          ...s,
          phase: 'deconstructing',
          total: d.total_steps,
          reference: `data:image/png;base64,${d.reference_b64}`,
          edges: `data:image/png;base64,${d.edges_b64}`,
          width: d.width,
          height: d.height,
          aspect: d.aspect || (d.width && d.height ? d.width / d.height : 1),
          finalSeed: d.seed_used,
        }));
        // Move from deconstruction → generating after a short cinematic delay.
        window.setTimeout(() => {
          setState((s) => (s.phase === 'deconstructing' ? { ...s, phase: 'generating' } : s));
        }, 1200);
      });

      es.addEventListener('step', (e) => {
        const d = JSON.parse((e as MessageEvent).data) as StepEvent;
        setState((s) => {
          const newPreviews = d.preview_b64
            ? [...s.previews, { index: d.index, dataUrl: `data:image/jpeg;base64,${d.preview_b64}` }]
            : s.previews;
          return {
            ...s,
            phase: s.phase === 'deconstructing' ? 'generating' : s.phase,
            step: d.index,
            total: d.total,
            previews: newPreviews,
            latestPreview: d.preview_b64
              ? `data:image/jpeg;base64,${d.preview_b64}`
              : s.latestPreview,
          };
        });
      });

      es.addEventListener('done', (e) => {
        const d = JSON.parse((e as MessageEvent).data) as DoneEvent;
        doneRef.current = true;
        // Close the stream immediately so the upcoming "stream closed" doesn't
        // surface as a transport error in our generic `error` handler.
        if (esRef.current) {
          esRef.current.close();
          esRef.current = null;
        }
        setState((s) => ({
          ...s,
          phase: 'revealing',
          finalImage: `data:image/png;base64,${d.image_b64}`,
          finalSeed: d.seed,
          elapsedS: d.elapsed_s,
        }));
        // Hold reveal animation, then mark complete.
        window.setTimeout(() => {
          setState((s) => (s.phase === 'revealing' ? { ...s, phase: 'complete' } : s));
          if (tickRef.current != null) {
            window.clearInterval(tickRef.current);
            tickRef.current = null;
          }
        }, 900);
      });

      es.addEventListener('error', (e) => {
        // After a successful `done`, the natural stream close fires `error` — suppress it.
        if (doneRef.current) return;

        // Try to read a structured error message; fall back to a generic one.
        let message = 'Stream interrupted before completion.';
        try {
          if ((e as MessageEvent).data) {
            const d = JSON.parse((e as MessageEvent).data) as ServerErrorEvent;
            if (d.message) message = d.message;
          }
        } catch {
          /* noop */
        }
        setState((s) => ({ ...s, phase: 'error', error: message }));
        stop();
      });
    },
    [stop],
  );

  useEffect(() => () => stop(), [stop]);

  return { state, start, stop, reset };
}
