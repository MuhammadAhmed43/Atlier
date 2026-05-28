import { useEffect, useState } from 'react';
import { fetchHealth } from '../api';
import type { Health } from '../types';

export function useHealth(intervalMs = 10000) {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const h = await fetchHealth();
        if (!cancelled) {
          setHealth(h);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(String((e as Error).message || e));
      }
    };

    tick();
    const id = window.setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return { health, error };
}
