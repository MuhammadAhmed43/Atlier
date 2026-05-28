import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import type { GenerationState } from '../hooks/useGeneration';

type Props = {
  state: GenerationState;
};

export function GenerationTheatre({ state }: Props) {
  const {
    phase,
    step,
    total,
    elapsedS,
    reference,
    edges,
    finalImage,
    latestPreview,
    previews,
    aspect,
  } = state;
  const stripRef = useRef<HTMLDivElement>(null);

  // Clamp aspect so a wildly tall or wide upload doesn't blow up the canvas.
  const safeAspect = Math.max(0.4, Math.min(2.5, aspect || 1));
  const canvasStyle = { aspectRatio: String(safeAspect) };

  // Auto-scroll preview strip to the newest frame
  useEffect(() => {
    if (stripRef.current) {
      stripRef.current.scrollLeft = stripRef.current.scrollWidth;
    }
  }, [previews.length]);

  if (phase === 'idle') return null;

  // Keep the dark theatre background for the reveal too — white labels on the
  // 3-up composition need a dark backdrop to be legible.
  const isDarkPhase =
    phase === 'deconstructing' ||
    phase === 'generating' ||
    phase === 'preparing' ||
    phase === 'revealing' ||
    phase === 'complete';

  const progressPct = total > 0 ? Math.min(100, (step / total) * 100) : 0;

  return (
    <section
      className={`w-full ${
        isDarkPhase
          ? 'bg-tertiary-container text-on-tertiary'
          : 'bg-background text-on-background'
      } py-16 px-margin-mobile md:px-margin-desktop flex flex-col items-center justify-center border-y border-on-tertiary-fixed-variant transition-colors duration-700`}
    >
      <div className="max-w-4xl w-full flex flex-col gap-8">
        {/* COMPLETE: Three-up editorial composition */}
        {phase === 'complete' && finalImage && (
          <ThreeUpReveal
            reference={reference}
            edges={edges}
            finalImage={finalImage}
            state={state}
            aspect={safeAspect}
          />
        )}

        {/* ACTIVE phases: dark theatre with main canvas */}
        {phase !== 'complete' && phase !== 'error' && (
          <>
            <div
              className="relative w-full max-w-[640px] mx-auto hairline-all border-on-tertiary-fixed-variant overflow-hidden bg-black"
              style={canvasStyle}
            >
              {/* Stage 1: Deconstruction — cross-fade ref → edges */}
              <AnimatePresence>
                {phase === 'preparing' && reference && (
                  <motion.img
                    key="pre-ref"
                    src={reference}
                    alt=""
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.4 }}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                )}
                {phase === 'deconstructing' && reference && (
                  <motion.img
                    key="dc-ref"
                    src={reference}
                    alt=""
                    initial={{ opacity: 1 }}
                    animate={{ opacity: 0, filter: 'blur(8px)', scale: 1.05 }}
                    transition={{ duration: 0.9, ease: 'easeInOut' }}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                )}
                {phase === 'deconstructing' && edges && (
                  <motion.img
                    key="dc-edge"
                    src={edges}
                    alt=""
                    initial={{ opacity: 0, scale: 1.1 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1.0, delay: 0.2, ease: 'easeOut' }}
                    className="absolute inset-0 w-full h-full object-contain mix-blend-screen animate-edge-glow"
                  />
                )}
              </AnimatePresence>

              {/* Stage 2/3: Generating — edge map behind, latest preview in front */}
              {phase === 'generating' && edges && (
                <img
                  src={edges}
                  alt=""
                  className="absolute inset-0 w-full h-full object-contain opacity-20 mix-blend-screen"
                />
              )}
              {phase === 'generating' && (
                <div className="absolute inset-0 noise-bg opacity-30 pointer-events-none" />
              )}
              {phase === 'generating' && latestPreview && (
                <motion.img
                  key={latestPreview}
                  src={latestPreview}
                  alt=""
                  initial={{ opacity: 0, scale: 1.05 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4 }}
                  className="absolute inset-0 w-full h-full object-cover"
                />
              )}

              {/* Stage 4: Reveal */}
              {phase === 'revealing' && finalImage && (
                <motion.img
                  key="reveal-final"
                  src={finalImage}
                  alt=""
                  initial={{ clipPath: 'inset(0 100% 0 0)' }}
                  animate={{ clipPath: 'inset(0 0% 0 0)' }}
                  transition={{ duration: 0.8, ease: [0.65, 0, 0.35, 1] }}
                  className="absolute inset-0 w-full h-full object-cover"
                />
              )}

              {/* Phase caption */}
              <div className="absolute inset-0 bg-gradient-to-t from-tertiary-container/80 to-transparent flex items-end p-6 pointer-events-none">
                <span className="font-headline-md text-headline-md text-on-tertiary opacity-90">
                  {phase === 'preparing' && 'Preparing canvas…'}
                  {phase === 'deconstructing' && 'Deconstructing into structure…'}
                  {phase === 'generating' && 'Weaving pixels…'}
                  {phase === 'revealing' && 'Reveal'}
                </span>
              </div>
            </div>

            {/* Progress row */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between font-mono-code text-mono-code text-primary-fixed-dim text-[11px]">
                <span>
                  Step {step.toString().padStart(2, '0')} / {total || '—'}
                </span>
                <span>{elapsedS.toFixed(2)}s elapsed</span>
              </div>
              <div className="w-full h-1 bg-on-tertiary-fixed-variant">
                <motion.div
                  className="h-full bg-secondary-container relative"
                  animate={{ width: `${progressPct}%` }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                >
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 bg-secondary rounded-full shadow-[0_0_8px_rgba(254,215,152,0.8)]" />
                </motion.div>
              </div>
            </div>

            {/* Preview timeline strip */}
            <div
              ref={stripRef}
              className="flex gap-2 overflow-x-auto pb-2 hide-scrollbar"
            >
              {Array.from({ length: Math.max(total, previews.length || 0, 20) }).map((_, i) => {
                const idx = i + 1;
                const frame = previews.find((p) => p.index === idx);
                const isActive = idx === step;
                const isPast = idx <= step;
                return (
                  <div
                    key={i}
                    className={`h-12 w-16 flex-shrink-0 bg-surface-tint relative overflow-hidden transition-opacity ${
                      isPast ? 'opacity-100' : 'opacity-20'
                    } ${isActive ? 'ring-1 ring-secondary' : ''}`}
                  >
                    {frame && (
                      <img
                        src={frame.dataUrl}
                        alt={`step ${idx}`}
                        className="w-full h-full object-cover"
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}

        {/* ERROR */}
        {phase === 'error' && (
          <div className="bg-error-container text-on-error-container p-6 hairline-all">
            <p className="font-headline-md text-headline-md mb-2">The atelier had a moment.</p>
            <p className="font-mono-code text-mono-code">{state.error}</p>
          </div>
        )}
      </div>
    </section>
  );
}

function ThreeUpReveal({
  reference,
  edges,
  finalImage,
  state,
  aspect,
}: {
  reference: string | null;
  edges: string | null;
  finalImage: string;
  state: GenerationState;
  aspect: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="flex flex-col gap-6"
    >
      <div className="grid grid-cols-3 gap-4">
        <PanelCell label="Reference" src={reference} aspect={aspect} />
        <PanelCell label="Structure" src={edges} invert aspect={aspect} />
        <PanelCell label="Generated" src={finalImage} highlight aspect={aspect} />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-4 font-mono-code text-mono-code text-on-tertiary-fixed-variant">
        <span>
          Seed: <span className="text-on-tertiary">{state.finalSeed}</span>
        </span>
        <span>
          Generated in <span className="text-on-tertiary">{state.elapsedS.toFixed(2)}s</span>
        </span>
        <a
          href={finalImage}
          download={`atelier-${state.finalSeed}.png`}
          className="font-label-sm text-label-sm uppercase tracking-widest text-on-tertiary border border-on-tertiary-fixed-variant px-4 py-2 hover:bg-on-tertiary hover:text-tertiary-container transition-colors"
        >
          Download
        </a>
      </div>
    </motion.div>
  );
}

function PanelCell({
  label,
  src,
  invert,
  highlight,
  aspect,
}: {
  label: string;
  src: string | null;
  invert?: boolean;
  highlight?: boolean;
  aspect: number;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div
        className={`hairline-all border-on-tertiary-fixed-variant bg-black overflow-hidden ${highlight ? 'ring-1 ring-secondary' : ''}`}
        style={{ aspectRatio: String(aspect) }}
      >
        {src && (
          <img
            src={src}
            alt={label}
            className={`w-full h-full object-cover ${invert ? 'mix-blend-screen' : ''}`}
          />
        )}
      </div>
      <span className="font-label-sm text-label-sm uppercase tracking-widest text-on-tertiary-fixed-variant">
        {label}
      </span>
    </div>
  );
}
