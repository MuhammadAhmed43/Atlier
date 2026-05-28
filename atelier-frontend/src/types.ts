export type Health = {
  device: string;
  dtype: string;
  pipeline_loaded: boolean;
  loading: boolean;
  lora: string | null;
  error: string | null;
  gpu: string | null;
  vram_total_gb: number | null;
};

export type SampleSummary = {
  id: string;
  url: string;
};

export type SampleList = {
  count: number;
  samples: SampleSummary[];
};

export type GenerationParams = {
  prompt: string;
  negative_prompt?: string;
  sample_id?: string;
  image_b64?: string;
  steps: number;
  guidance: number;
  canny_low?: number;
  canny_high?: number;
  seed: number;
  preview_every?: number;
};

export type ReadyEvent = {
  reference_b64: string;
  edges_b64: string;
  total_steps: number;
  seed_used: number;
  width: number;
  height: number;
  aspect: number;
};

export type StepEvent = {
  index: number;
  total: number;
  preview_b64?: string;
  preview_error?: string;
};

export type DoneEvent = {
  image_b64: string;
  seed: number;
  elapsed_s: number;
};

export type ErrorEvent = {
  message: string;
};

export type GenerationPhase =
  | 'idle'
  | 'preparing'
  | 'deconstructing'
  | 'generating'
  | 'revealing'
  | 'complete'
  | 'error';

export type LookbookEntry = {
  id: string;
  prompt: string;
  category?: string;
  seed: number;
  steps: number;
  guidance: number;
  aspect: number;
  imageDataUrl: string;
  edgesDataUrl: string;
  referenceDataUrl: string;
  createdAt: number;
};
