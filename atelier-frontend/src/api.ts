import type { GenerationParams, Health, SampleList } from './types';

export const API_BASE: string = import.meta.env.VITE_API_BASE || 'http://localhost:8001';

export async function fetchHealth(): Promise<Health> {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function fetchSamples(): Promise<SampleList> {
  const r = await fetch(`${API_BASE}/samples`);
  if (!r.ok) throw new Error(`samples ${r.status}`);
  return r.json();
}

export function sampleUrl(filename: string): string {
  return `${API_BASE}/samples/${filename}`;
}

/** Build the SSE URL for /generate/stream. We use GET (EventSource limitation). */
export function buildStreamUrl(p: GenerationParams): string {
  const q = new URLSearchParams();
  q.set('prompt', p.prompt);
  if (p.negative_prompt) q.set('negative_prompt', p.negative_prompt);
  if (p.sample_id) q.set('sample_id', p.sample_id);
  if (p.image_b64) q.set('image_b64', p.image_b64);
  q.set('steps', String(p.steps));
  q.set('guidance', String(p.guidance));
  if (p.canny_low != null) q.set('canny_low', String(p.canny_low));
  if (p.canny_high != null) q.set('canny_high', String(p.canny_high));
  q.set('seed', String(p.seed));
  if (p.preview_every != null) q.set('preview_every', String(p.preview_every));
  return `${API_BASE}/generate/stream?${q.toString()}`;
}
