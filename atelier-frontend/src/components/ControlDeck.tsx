import { CATEGORIES, PRESETS, randomPreset } from '../presets';

export type DeckState = {
  category: string;
  preset: string;
  prompt: string;
  steps: number;
  guidance: number;
  seed: number;
};

type Props = {
  state: DeckState;
  onChange: (next: DeckState) => void;
  onGenerate: () => void;
  busy: boolean;
  ready: boolean;
};

export function ControlDeck({ state, onChange, onGenerate, busy, ready }: Props) {
  const set = <K extends keyof DeckState>(key: K, value: DeckState[K]) =>
    onChange({ ...state, [key]: value });

  const onCategory = (cat: string) => {
    const presets = PRESETS[cat];
    const firstName = Object.keys(presets)[0];
    onChange({
      ...state,
      category: cat,
      preset: firstName,
      prompt: presets[firstName],
    });
  };

  const onPreset = (name: string) => {
    onChange({ ...state, preset: name, prompt: PRESETS[state.category][name] });
  };

  const onSurprise = () => {
    const r = randomPreset();
    onChange({ ...state, category: r.category, preset: r.name, prompt: r.prompt });
  };

  return (
    <div className="flex flex-col gap-8">
      <div>
        <Selector
          label="Category"
          value={state.category}
          options={CATEGORIES}
          onChange={onCategory}
        />
        <Selector
          label="Preset"
          value={state.preset}
          options={Object.keys(PRESETS[state.category] || {})}
          onChange={onPreset}
        />

        <div className="relative mt-4">
          <textarea
            value={state.prompt}
            onChange={(e) => set('prompt', e.target.value)}
            placeholder="Describe the garment you want to generate…"
            className="w-full h-32 bg-surface-container-lowest hairline-all p-4 font-mono-code text-mono-code focus:outline-none focus:border-primary resize-none"
          />
          <button
            type="button"
            onClick={onSurprise}
            className="absolute bottom-3 right-3 font-label-sm text-label-sm uppercase px-3 py-1 bg-surface-variant hover:bg-surface-container-high transition-colors"
          >
            Surprise Me
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <h3 className="font-label-sm text-label-sm uppercase tracking-widest text-on-surface-variant hairline-b pb-2">
          Advanced Controls
        </h3>

        <Slider
          label="Quality Steps"
          value={state.steps}
          min={10}
          max={40}
          step={1}
          onChange={(v) => set('steps', v)}
          renderValue={(v) => v}
        />

        <Slider
          label="Prompt Strength"
          value={state.guidance}
          min={1}
          max={15}
          step={0.5}
          onChange={(v) => set('guidance', v)}
          renderValue={(v) => v.toFixed(1)}
        />

        <div className="flex flex-col gap-2 mt-2">
          <label className="font-label-sm text-label-sm uppercase">Seed</label>
          <div className="flex gap-2">
            <input
              type="number"
              value={state.seed}
              onChange={(e) => set('seed', Number(e.target.value))}
              className="flex-1 bg-surface-container-lowest hairline-all px-3 py-2 font-mono-code text-mono-code focus:outline-none focus:border-primary"
            />
            <button
              type="button"
              onClick={() => set('seed', -1)}
              className="font-label-sm text-label-sm uppercase px-3 py-2 hairline-all hover:bg-surface-variant"
              title="Random seed"
            >
              −1
            </button>
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={onGenerate}
        disabled={busy || !ready}
        className="w-full bg-secondary text-on-secondary disabled:bg-surface-variant disabled:text-on-surface-variant font-label-sm text-label-sm uppercase tracking-widest py-4 mt-auto hover:bg-secondary-fixed-dim transition-colors relative group overflow-hidden"
      >
        {busy ? 'Weaving Pixels…' : ready ? 'Generate' : 'Backend Offline'}
        <div className="absolute bottom-0 left-0 w-full h-[2px] bg-secondary-fixed-dim transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-300" />
      </button>
    </div>
  );
}

function Selector({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex justify-between items-end mb-2">
      <span className="font-label-sm text-label-sm uppercase text-on-surface-variant tracking-widest">
        {label}
      </span>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="font-mono-code text-mono-code text-primary bg-transparent appearance-none pr-5 cursor-pointer focus:outline-none"
        >
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
        <span className="pointer-events-none absolute right-0 top-1/2 -translate-y-1/2 material-symbols-outlined text-[14px] text-on-surface-variant">
          expand_more
        </span>
      </div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  renderValue,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  renderValue: (v: number) => string | number;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between">
        <label className="font-label-sm text-label-sm uppercase">{label}</label>
        <span className="font-mono-code text-mono-code">{renderValue(value)}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-primary h-1 bg-surface-variant appearance-none"
      />
      <div className="flex justify-between text-[10px] uppercase tracking-widest text-on-surface-variant mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
