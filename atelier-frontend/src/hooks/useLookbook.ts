import { useCallback, useEffect, useState } from 'react';
import type { LookbookEntry } from '../types';

const STORAGE_KEY = 'atelier:lookbook:v1';
const MAX_ENTRIES = 24;

function load(): LookbookEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as LookbookEntry[]) : [];
  } catch {
    return [];
  }
}

function save(entries: LookbookEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    /* quota exceeded — silently drop */
  }
}

export function useLookbook() {
  const [entries, setEntries] = useState<LookbookEntry[]>(() => load());

  useEffect(() => {
    save(entries);
  }, [entries]);

  const add = useCallback((entry: Omit<LookbookEntry, 'id' | 'createdAt'>) => {
    const newEntry: LookbookEntry = {
      ...entry,
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      createdAt: Date.now(),
    };
    setEntries((prev) => [newEntry, ...prev].slice(0, MAX_ENTRIES));
    return newEntry;
  }, []);

  const remove = useCallback((id: string) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return { entries, add, remove, clear };
}
