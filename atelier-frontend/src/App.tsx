import { useEffect, useRef, useState } from 'react';
import { TopBar } from './components/TopBar';
import { SamplePicker } from './components/SamplePicker';
import { ControlDeck, type DeckState } from './components/ControlDeck';
import { GenerationTheatre } from './components/GenerationTheatre';
import { LookbookArchive } from './components/LookbookArchive';
import { AboutSection } from './components/AboutSection';
import { Footer } from './components/Footer';
import { useGeneration } from './hooks/useGeneration';
import { useHealth } from './hooks/useHealth';
import { useLookbook } from './hooks/useLookbook';
import { PRESETS } from './presets';

type SelectedRef = { sampleId?: string; uploadDataUrl?: string };

const initialDeck: DeckState = {
  category: 'Outerwear',
  preset: 'Navy Wool Blazer',
  prompt: PRESETS.Outerwear['Navy Wool Blazer'],
  steps: 20,
  guidance: 7.5,
  seed: -1,
};

export default function App() {
  const { health, error: healthError } = useHealth();
  const { state: gen, start, reset } = useGeneration();
  const { entries, add, remove } = useLookbook();

  const [deck, setDeck] = useState<DeckState>(initialDeck);
  const [selected, setSelected] = useState<SelectedRef | null>(null);
  const theatreRef = useRef<HTMLDivElement>(null);

  const busy =
    gen.phase === 'preparing' ||
    gen.phase === 'deconstructing' ||
    gen.phase === 'generating' ||
    gen.phase === 'revealing';

  const ready = !!health?.pipeline_loaded;

  const handleGenerate = () => {
    if (!selected) return;
    const prompt = deck.prompt.trim();
    if (!prompt) return;
    start({
      prompt,
      sample_id: selected.sampleId,
      image_b64: selected.uploadDataUrl
        ? selected.uploadDataUrl.split(',')[1]
        : undefined,
      steps: deck.steps,
      guidance: deck.guidance,
      seed: deck.seed,
      preview_every: 2,
    });
    // Scroll handled by the useEffect below once `phase` flips to `preparing`.
  };

  // Smoothly scroll the theatre into view as soon as a generation kicks off.
  useEffect(() => {
    if (gen.phase === 'preparing') {
      // Defer one frame so the theatre actually mounts before we scroll to it.
      requestAnimationFrame(() => {
        theatreRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
  }, [gen.phase]);

  // When generation completes, archive to the lookbook.
  useEffect(() => {
    if (gen.phase === 'complete' && gen.finalImage && gen.reference && gen.edges) {
      add({
        prompt: deck.prompt,
        category: deck.category,
        seed: gen.finalSeed ?? deck.seed,
        steps: deck.steps,
        guidance: deck.guidance,
        aspect: gen.aspect,
        imageDataUrl: gen.finalImage,
        edgesDataUrl: gen.edges,
        referenceDataUrl: gen.reference,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gen.phase]);

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar health={health} healthError={healthError} generation={gen} />

      <main className="flex-grow flex flex-col">
        {/* Section 1: Studio */}
        <section
          id="studio"
          className="grid grid-cols-1 md:grid-cols-12 gap-gutter px-margin-mobile md:px-margin-desktop py-12 scroll-mt-20"
        >
          <div className="md:col-span-7 flex flex-col gap-6">
            <SamplePicker selected={selected} onSelect={setSelected} />
          </div>
          <div className="md:col-span-5 flex flex-col md:sticky md:top-28 self-start">
            <ControlDeck
              state={deck}
              onChange={setDeck}
              onGenerate={handleGenerate}
              busy={busy}
              ready={ready}
            />
          </div>
        </section>

        {/* Section 2: Generation Theatre */}
        <div ref={theatreRef} className="scroll-mt-20">
          <GenerationTheatre state={gen} />
        </div>

        {gen.phase === 'complete' && (
          <div className="bg-background py-6 flex justify-center">
            <button
              type="button"
              onClick={reset}
              className="font-label-sm text-label-sm uppercase tracking-widest px-8 py-3 border border-primary text-primary hover:bg-primary hover:text-on-primary transition-colors"
            >
              Clear Theatre
            </button>
          </div>
        )}

        {/* Section 3: Lookbook */}
        <LookbookArchive entries={entries} onRemove={remove} />

        {/* Section 4: About */}
        <AboutSection health={health} />
      </main>

      <Footer health={health} />
    </div>
  );
}
