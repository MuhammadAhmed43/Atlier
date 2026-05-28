import type { Health } from '../types';
import type { GenerationState } from '../hooks/useGeneration';

type Props = {
  health: Health | null;
  healthError: string | null;
  generation: GenerationState;
};

export function TopBar({ health, healthError, generation }: Props) {
  let statusLabel: string;
  let statusOk = true;

  if (healthError) {
    statusLabel = 'Backend • Offline';
    statusOk = false;
  } else if (!health) {
    statusLabel = 'Backend • Connecting…';
  } else if (!health.pipeline_loaded) {
    statusLabel = health.loading ? 'Model • Loading…' : 'Model • Not Ready';
    statusOk = false;
  } else if (generation.phase === 'generating' || generation.phase === 'deconstructing') {
    statusLabel = `Generating • ${generation.step}/${generation.total || '?'}`;
  } else if (generation.phase === 'preparing') {
    statusLabel = 'Preparing…';
  } else {
    statusLabel = 'Model • Ready';
  }

  return (
    <header className="bg-background border-b border-outline-variant sticky top-0 z-50 flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop h-20">
      <div className="flex items-center gap-12">
        <h1 className="font-headline-md text-headline-md font-normal tracking-tight text-primary">
          ATELIER
        </h1>
        <nav className="hidden md:flex gap-8">
          <a
            href="#studio"
            className="font-label-sm text-label-sm text-primary border-b border-primary pb-1 uppercase"
          >
            Studio
          </a>
          <a
            href="#lookbook"
            className="font-label-sm text-label-sm text-on-surface-variant hover:text-primary transition-colors duration-300 pb-1 uppercase"
          >
            Lookbook
          </a>
          <a
            href="#about"
            className="font-label-sm text-label-sm text-on-surface-variant hover:text-primary transition-colors duration-300 pb-1 uppercase"
          >
            About
          </a>
        </nav>
      </div>
      <div className="flex items-center gap-4">
        <div
          className="flex items-center gap-2 font-mono-code text-mono-code"
          style={{ color: statusOk ? '#765a26' : '#ba1a1a' }}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              statusOk
                ? 'bg-secondary animate-pulse-gold'
                : 'bg-error'
            }`}
          />
          {statusLabel}
        </div>
      </div>
    </header>
  );
}
