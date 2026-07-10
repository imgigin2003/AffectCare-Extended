import { useEffect, useState } from 'react';

// Persists the light/dark choice and reflects it on <html data-theme>.
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('affectcare-theme');
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('affectcare-theme', theme);
  }, [theme]);

  const toggle = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));

  return { theme, toggle };
}
