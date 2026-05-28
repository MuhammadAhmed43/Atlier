import type { LookbookEntry } from '../types';

type Props = {
  entries: LookbookEntry[];
  onRemove: (id: string) => void;
};

export function LookbookArchive({ entries, onRemove }: Props) {
  if (entries.length === 0) {
    return (
      <section
        id="lookbook"
        className="px-margin-mobile md:px-margin-desktop py-16 flex flex-col gap-12"
      >
        <h2 className="font-headline-lg text-headline-lg-mobile md:text-headline-lg text-primary text-center">
          Lookbook Archive
        </h2>
        <p className="text-center text-on-surface-variant max-w-lg mx-auto">
          Pieces you create will collect here. Pick a silhouette above and tell us what to weave —
          the first one's always free.
        </p>
      </section>
    );
  }

  return (
    <section
      id="lookbook"
      className="px-margin-mobile md:px-margin-desktop py-16 flex flex-col gap-12 scroll-mt-20"
    >
      <div className="flex items-end justify-between">
        <h2 className="font-headline-lg text-headline-lg-mobile md:text-headline-lg text-primary">
          Lookbook Archive
        </h2>
        <span className="font-mono-code text-mono-code text-on-surface-variant">
          {entries.length} piece{entries.length === 1 ? '' : 's'}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-gutter">
        {entries.map((e, i) => (
          <article
            key={e.id}
            className={`flex flex-col gap-4 group ${i % 3 === 1 ? 'md:mt-12' : ''}`}
          >
            <div
              className="hairline-all overflow-hidden bg-surface-variant relative"
              style={{ aspectRatio: String(Math.max(0.4, Math.min(2.5, e.aspect || 0.75))) }}
            >
              <img
                src={e.imageDataUrl}
                alt={e.prompt}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700 ease-out"
              />
              <button
                type="button"
                onClick={() => onRemove(e.id)}
                className="absolute top-2 right-2 w-7 h-7 flex items-center justify-center bg-background/80 text-on-surface-variant opacity-0 group-hover:opacity-100 hover:text-error transition-all"
                title="Remove"
              >
                <span className="material-symbols-outlined text-[18px]">close</span>
              </button>
            </div>
            <div>
              <p className="font-headline-md text-primary italic text-[22px] leading-tight line-clamp-2">
                “{shortName(e.prompt)}”
              </p>
              <p className="font-mono-code text-mono-code text-on-surface-variant text-[11px] mt-2">
                Seed: {e.seed} • CFG: {e.guidance.toFixed(1)} • {e.steps} steps
              </p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function shortName(prompt: string): string {
  // Take first ~6 words, trim trailing comma/dot.
  const words = prompt.split(/\s+/).slice(0, 7).join(' ');
  return words.replace(/[,.;:]+$/, '');
}
