import type { Health } from '../types';

export function Footer({ health }: { health: Health | null }) {
  const model = health?.lora || 'fashion-lora';
  const gpu = health?.gpu || '—';
  return (
    <footer className="bg-background border-t border-outline-variant flex flex-col md:flex-row justify-between items-center w-full px-margin-mobile md:px-margin-desktop py-6 gap-4 mt-auto">
      <div className="font-label-sm text-label-sm uppercase tracking-widest text-on-surface-variant text-center md:text-left">
        Stable Diffusion v1.5 • ControlNet (Canny) • LoRA — Fine-tuned on FashionGen
      </div>
      <div className="font-mono-code text-mono-code text-on-surface-variant flex flex-wrap gap-x-6 gap-y-1 justify-center">
        <span>Model: {model}</span>
        <span>GPU: {gpu}</span>
      </div>
    </footer>
  );
}
