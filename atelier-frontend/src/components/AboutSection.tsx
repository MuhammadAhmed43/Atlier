import type { Health } from '../types';

type Props = {
  health: Health | null;
};

export function AboutSection({ health }: Props) {
  const gpu = health?.gpu ?? 'NVIDIA GPU';
  const dtype = health?.dtype?.replace('torch.', '') ?? 'float16';
  const vram = health?.vram_total_gb ? `${health.vram_total_gb} GB` : '—';

  return (
    <section
      id="about"
      className="bg-surface-container-low border-y border-outline-variant px-margin-mobile md:px-margin-desktop py-20 md:py-24 scroll-mt-20"
    >
      <div className="max-w-[1100px] mx-auto flex flex-col gap-16">
        {/* Eyebrow + headline */}
        <div className="flex flex-col gap-4 max-w-3xl">
          <span className="font-label-sm text-label-sm uppercase tracking-[0.3em] text-on-surface-variant">
            About the Atelier
          </span>
          <h2 className="font-headline-lg text-headline-lg-mobile md:text-headline-lg text-primary leading-tight">
            A controllable diffusion studio,{' '}
            <span className="italic text-secondary">tailored for fashion.</span>
          </h2>
          <p className="font-body-lg text-body-lg text-on-surface-variant max-w-2xl mt-2">
            ATELIER turns a reference photo and a written brief into a brand-new garment that
            preserves the silhouette of the original — built on Stable Diffusion v1.5, conditioned
            with ControlNet, and fine-tuned on the fashion domain with a Low-Rank Adapter.
          </p>
        </div>

        {/* Stat row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-outline-variant hairline-all">
          <Stat label="Base Model" value="SD v1.5" sub="860M params · frozen" />
          <Stat label="Conditioning" value="ControlNet" sub="Canny · 1.4B params · frozen" />
          <Stat label="Fine-tune" value="LoRA r=4" sub="≈4M trainable params" />
          <Stat label="Training Set" value="30k images" sub="FashionGen · 256²" />
        </div>

        {/* Pipeline diagram */}
        <div className="flex flex-col gap-6">
          <h3 className="font-headline-md text-headline-md text-primary">The pipeline</h3>
          <pre className="font-mono-code text-mono-code overflow-x-auto bg-surface-container-lowest hairline-all p-6 leading-relaxed text-on-surface">
{`     ┌──────────────────────┐        ┌──────────────────────┐
     │  Reference image     │        │   Text prompt        │
     │  (your upload or     │        │  "navy wool blazer   │
     │   a sample garment)  │        │   with gold buttons" │
     └──────────┬───────────┘        └──────────┬───────────┘
                │                               │
                ▼                               ▼
     ┌──────────────────────┐        ┌──────────────────────┐
     │  Canny edge detector │        │   CLIP text encoder  │
     │  → structural map    │        │   → 77 × 768 embeds  │
     └──────────┬───────────┘        └──────────┬───────────┘
                │                               │
                ▼                               ▼
     ┌────────────────────────────────────────────────────────┐
     │           Stable Diffusion U-Net  +  LoRA              │
     │       (frozen)                  (4M trainable)         │
     │                                                        │
     │   ControlNet branch injects edge features at every     │
     │   block, locking the silhouette of the generated       │
     │   garment to the reference image.                      │
     └────────────────────────┬───────────────────────────────┘
                              │
                              ▼
     ┌────────────────────────────────────────────────────────┐
     │            VAE decoder  →  pixel-space garment         │
     └────────────────────────────────────────────────────────┘`}
          </pre>
        </div>

        {/* Two-column: how the theatre works + training details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <Block title="How the live theatre works">
            <p>
              Every denoising step the U-Net runs, the live{' '}
              <em className="text-primary">latent</em> tensor is intercepted by a callback,
              decoded through the VAE, and pushed to your browser as a base64 JPEG over a{' '}
              <strong className="text-primary">Server-Sent Events</strong> stream.
            </p>
            <p>
              That's why you can see the garment cohering in real time — the timeline thumbnails
              are not pre-rendered. Each frame is an honest decode of what the model is{' '}
              <em className="text-primary">actually</em> thinking at step <code>k</code>.
            </p>
            <p className="font-mono-code text-mono-code text-on-surface-variant">
              event: ready &nbsp; → reference + edges<br />
              event: step &nbsp; → preview every 2 steps<br />
              event: done &nbsp; → final 256² (or aspect-matched) image
            </p>
          </Block>

          <Block title="Training specifications">
            <SpecRow label="Resolution" value="256 × 256" />
            <SpecRow label="Optimizer" value="AdamW · LR 1e-4" />
            <SpecRow label="Schedule" value="Cosine · 500-step warmup" />
            <SpecRow label="Steps" value="5,000" />
            <SpecRow label="Effective batch" value="4 (2 × grad-accum 2)" />
            <SpecRow label="Precision" value="FP32 (ROCm/Kaggle parity)" />
            <SpecRow label="LoRA rank / alpha" value="4 / 4" />
            <SpecRow label="Seed" value="42" />
            <SpecRow label="Checkpoint every" value="1,000 steps" />
          </Block>
        </div>

        {/* Evaluation */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-outline-variant hairline-all">
          <Stat label="FID" value="quality" sub="lower is better" />
          <Stat label="KID" value="diversity" sub="robust on small sets" />
          <Stat label="LPIPS" value="perceptual" sub="AlexNet features" />
          <Stat label="CLIP" value="text↔image" sub="higher is better" />
        </div>

        {/* Live runtime card */}
        <div className="hairline-all p-6 bg-surface-container-lowest flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <span className="font-label-sm text-label-sm uppercase tracking-widest text-on-surface-variant block mb-1">
              Live Runtime
            </span>
            <p className="font-headline-md text-[24px] text-primary">{gpu}</p>
          </div>
          <div className="grid grid-cols-3 gap-6 font-mono-code text-mono-code">
            <div>
              <div className="text-on-surface-variant text-[11px] uppercase tracking-widest">Precision</div>
              <div className="text-primary">{dtype}</div>
            </div>
            <div>
              <div className="text-on-surface-variant text-[11px] uppercase tracking-widest">VRAM</div>
              <div className="text-primary">{vram}</div>
            </div>
            <div>
              <div className="text-on-surface-variant text-[11px] uppercase tracking-widest">LoRA</div>
              <div className="text-primary truncate">{health?.lora ?? '—'}</div>
            </div>
          </div>
        </div>

        {/* Credits */}
        <div className="flex flex-col gap-4">
          <h3 className="font-headline-md text-headline-md text-primary">Acknowledgements</h3>
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-2 text-on-surface-variant">
            <Credit who="Stability AI" what="Stable Diffusion v1.5" />
            <Credit who="lllyasviel" what="ControlNet (Canny) v1.1p" />
            <Credit who="Microsoft Research" what="LoRA: Low-Rank Adaptation" />
            <Credit who="Hugging Face" what="diffusers · transformers · accelerate" />
            <Credit who="FashionGen" what="Garment image / caption pairs" />
            <Credit who="Unsplash photographers" what="Reference imagery in the sample picker" />
          </ul>
          <p className="font-mono-code text-mono-code text-on-surface-variant mt-4 text-[11px] uppercase tracking-widest">
            ATELIER · A deep-learning course project · v0.1
          </p>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="bg-surface-container-lowest p-6 flex flex-col gap-1">
      <span className="font-label-sm text-label-sm uppercase tracking-widest text-on-surface-variant">
        {label}
      </span>
      <span className="font-headline-md text-[28px] text-primary leading-tight">{value}</span>
      <span className="font-mono-code text-mono-code text-on-surface-variant text-[11px]">{sub}</span>
    </div>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-4">
      <h3 className="font-headline-md text-headline-md text-primary hairline-b pb-3">{title}</h3>
      <div className="flex flex-col gap-3 text-on-surface-variant leading-relaxed">{children}</div>
    </div>
  );
}

function SpecRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-baseline border-b border-outline-variant pb-2">
      <span className="font-label-sm text-label-sm uppercase tracking-widest text-on-surface-variant">
        {label}
      </span>
      <span className="font-mono-code text-mono-code text-primary text-right">{value}</span>
    </div>
  );
}

function Credit({ who, what }: { who: string; what: string }) {
  return (
    <li className="flex justify-between items-baseline gap-4 border-b border-outline-variant pb-2">
      <span className="font-mono-code text-mono-code text-primary">{who}</span>
      <span className="font-mono-code text-mono-code text-on-surface-variant text-right text-[11px]">
        {what}
      </span>
    </li>
  );
}
