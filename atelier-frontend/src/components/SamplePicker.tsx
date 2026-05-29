import { useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE, fetchSamples } from '../api';
import type { SampleSummary } from '../types';

type Props = {
  selected: { sampleId?: string; uploadDataUrl?: string } | null;
  onSelect: (next: { sampleId?: string; uploadDataUrl?: string }) => void;
};

const STRIP_SIZE = 10;

function pickRandom<T>(arr: T[], n: number): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, Math.min(n, copy.length));
}

function pickEvenly<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return arr;
  const step = Math.max(1, Math.floor(arr.length / n));
  return Array.from({ length: n }, (_, i) => arr[i * step]).filter(Boolean);
}

export function SamplePicker({ selected, onSelect }: Props) {
  const [all, setAll] = useState<SampleSummary[]>([]);
  const [strip, setStrip] = useState<SampleSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    fetchSamples()
      .then((s) => {
        if (cancelled) return;
        setAll(s.samples);
        const initial = pickEvenly(s.samples, STRIP_SIZE);
        setStrip(initial);
        if (!selected && initial[0]) onSelect({ sampleId: initial[0].id });
      })
      .catch((e) => {
        if (!cancelled) setError(String(e.message || e));
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedSampleId = selected?.sampleId;

  const heroUrl = useMemo(() => {
    if (selected?.uploadDataUrl) return selected.uploadDataUrl;
    const found = all.find((s) => s.id === selectedSampleId);
    return found ? `${API_BASE}${found.url}` : null;
  }, [selected, selectedSampleId, all]);

  const currentIdx = strip.findIndex((s) => s.id === selectedSampleId);

  const goPrev = () => {
    if (strip.length === 0) return;
    const next = strip[(currentIdx <= 0 ? strip.length : currentIdx) - 1];
    if (next) onSelect({ sampleId: next.id });
  };
  const goNext = () => {
    if (strip.length === 0) return;
    const next = strip[(currentIdx + 1) % strip.length];
    if (next) onSelect({ sampleId: next.id });
  };
  const goRandom = () => {
    if (strip.length === 0) return;
    const next = strip[Math.floor(Math.random() * strip.length)];
    if (next) onSelect({ sampleId: next.id });
  };
  const shuffleStrip = () => {
    const next = pickRandom(all, STRIP_SIZE);
    setStrip(next);
    if (next[0]) onSelect({ sampleId: next[0].id });
  };

  const handleUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        onSelect({ uploadDataUrl: reader.result });
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="hairline-all p-4 bg-surface-container-lowest">
        <div className="w-full aspect-square bg-surface-variant flex items-center justify-center overflow-hidden">
          {heroUrl ? (
            <img
              src={heroUrl}
              alt="Reference"
              className="w-full h-full object-cover"
              style={{ imageRendering: 'auto' }}
            />
          ) : error ? (
            <p className="font-mono-code text-mono-code text-error px-6 text-center">
              Couldn’t reach backend: {error}
            </p>
          ) : (
            <p className="font-label-sm text-label-sm uppercase tracking-widest text-on-surface-variant">
              Loading samples…
            </p>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex gap-2">
            <NavButton onClick={goPrev}>Previous</NavButton>
            <NavButton onClick={goRandom}>Random</NavButton>
            <NavButton onClick={goNext}>Next</NavButton>
          </div>
          <NavButton onClick={shuffleStrip}>
            <span className="material-symbols-outlined text-[16px] align-middle mr-1">shuffle</span>
            Shuffle Set
          </NavButton>
        </div>

        <div className="flex gap-4 overflow-x-auto pb-4 hide-scrollbar">
          {strip.map((s) => {
            const url = `${API_BASE}${s.url}`;
            const isSelected = s.id === selectedSampleId && !selected?.uploadDataUrl;
            return (
              <button
                key={s.id}
                onClick={() => onSelect({ sampleId: s.id })}
                title={s.id}
                className={`w-24 h-24 flex-shrink-0 hairline-all bg-surface-variant bg-cover bg-center transition-all ${
                  isSelected ? 'outline outline-2 outline-primary' : 'opacity-80 hover:opacity-100'
                }`}
                style={{ backgroundImage: `url(${url})` }}
              />
            );
          })}
          <button
            onClick={() => fileRef.current?.click()}
            className={`w-24 h-24 flex-shrink-0 hairline-all border-dashed border-2 flex items-center justify-center cursor-pointer hover:bg-surface-variant transition-colors text-on-surface-variant ${
              selected?.uploadDataUrl ? 'outline outline-2 outline-primary' : ''
            }`}
            title="Upload your own"
          >
            <span className="material-symbols-outlined">upload</span>
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleUpload(f);
            }}
          />
        </div>
      </div>
    </div>
  );
}

function NavButton({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="font-label-sm text-label-sm uppercase px-4 py-2 hairline-all hover:bg-surface-variant transition-colors flex items-center"
    >
      {children}
    </button>
  );
}
