import os

ROOT = r"c:\MyFiles\Desktop\MargDrishti\frontend"

def write(path, content):
    full_path = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

write("vite.config.js", """
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
    }
  }
})
""")

write("src/index.css", """
@import 'tailwindcss';

@theme {
  --color-cream-50: #FFFDF3;
  --color-cream-100: #FFF9DB;
  --color-cream-200: #F4ECC6;
  --color-lavender-100: #EEEDFB;
  --color-lavender-200: #CDCBF0;
  --color-lavender-300: #B3ACDF;
  --color-purple-400: #9479B8;
  --color-purple-500: #7C5FA6;
  --color-purple-700: #4E3A73;
  --color-ink-900: #2A2438;
  --color-ink-600: #5B5470;
  --color-ink-400: #8E88A0;
  --color-sage: #6F9E86;
  --color-rose: #C0707F;
  --color-ochre: #C39A45;
  
  --font-display: 'Cormorant Garamond', serif;
  --font-sans: 'DM Sans', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
  --font-deva: 'Tiro Devanagari Hindi', serif;

  --shadow-soft: 0 1px 2px rgba(78,58,115,0.06), 0 8px 24px rgba(78,58,115,0.06);
  --shadow-lift: 0 4px 6px rgba(78,58,115,0.06), 0 12px 32px rgba(78,58,115,0.08);

  --ease-soft: cubic-bezier(0.22, 1, 0.36, 1);
}

@layer base {
  body {
    @apply bg-cream-50 text-ink-900 font-sans text-base antialiased selection:bg-lavender-200;
    line-height: 1.65;
  }
  h1, h2, h3, h4, h5, h6 {
    @apply font-display tracking-tight text-ink-900;
  }
  h1 {
    font-size: clamp(3rem, 5vw, 4rem); /* ~48px-64px */
  }
  h2 {
    font-size: clamp(2rem, 3vw, 2.5rem); /* ~32px-40px */
  }
  :focus-visible {
    @apply outline-purple-500 outline-2 outline-offset-2;
  }
  html {
    scroll-behavior: smooth;
  }
  .tabular-nums {
    font-variant-numeric: tabular-nums;
  }
}
""")

write("src/main.jsx", """
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

import '@fontsource/cormorant-garamond/600.css';
import '@fontsource/dm-sans/400.css';
import '@fontsource/dm-sans/500.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/tiro-devanagari-hindi/400.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""")

write("src/App.jsx", """
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './app/Layout';
import Home from './pages/Home';
import Dataset from './pages/Dataset';
import Models from './pages/Models';
import LiveDemo from './pages/LiveDemo';
import Comparison from './pages/Comparison';
import About from './pages/About';
import NotFound from './pages/NotFound';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="dataset" element={<Dataset />} />
          <Route path="models" element={<Models />} />
          <Route path="demo" element={<LiveDemo />} />
          <Route path="comparison" element={<Comparison />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
""")

write("src/app/Layout.jsx", """
import React, { useState, useEffect } from 'react';
import { Outlet, NavLink, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { Menu, X } from 'lucide-react';
import { PageTransition } from '../lib/motion';

function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2 group outline-none focus-visible:ring-2 focus-visible:ring-purple-500 rounded px-1 -ml-1">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-purple-500">
        <path d="M12 2L2 20H22L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
        <path d="M12 16C12 16 9 12 12 10C15 12 12 16 12 16Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
      </svg>
      <div className="flex items-baseline gap-1.5">
        <span className="font-display font-semibold text-xl tracking-tight text-ink-900 group-hover:text-purple-700 transition-colors">MargDrishti</span>
        <span className="font-deva text-purple-500 text-sm">मार्गदृष्टि</span>
      </div>
    </Link>
  );
}

function NavItem({ to, children }) {
  return (
    <NavLink to={to} className={({ isActive }) => `relative px-3 py-2 text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-purple-500 rounded transition-colors ${isActive ? 'text-purple-700' : 'text-ink-600 hover:text-ink-900'}`}>
      {({ isActive }) => (
        <>
          {children}
          {isActive && (
            <motion.div
              layoutId="nav-indicator"
              className="absolute bottom-0 left-3 right-3 h-[2px] bg-purple-500 rounded-t-full"
              transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
            />
          )}
        </>
      )}
    </NavLink>
  );
}

export default function Layout() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState('unknown');

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    fetch('/api/health')
      .then(res => {
        if (res.ok) setApiStatus('ok');
        else setApiStatus('error');
      })
      .catch(() => setApiStatus('error'));
  }, []);

  const navLinks = [
    { to: '/', label: 'Home' },
    { to: '/dataset', label: 'Dataset' },
    { to: '/models', label: 'Models' },
    { to: '/demo', label: 'Live Demo' },
    { to: '/comparison', label: 'Comparison' },
    { to: '/about', label: 'About' },
  ];

  return (
    <div className="min-h-screen flex flex-col selection:bg-lavender-200">
      <header className={`sticky top-0 z-50 transition-all duration-300 ${scrolled ? 'bg-cream-50/80 backdrop-blur-md border-b border-lavender-200 shadow-soft' : 'bg-transparent border-b border-transparent'}`}>
        <div className="max-w-[1120px] mx-auto px-4 md:px-8 h-16 flex items-center justify-between">
          <Logo />
          
          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map(link => (
              <NavItem key={link.to} to={link.to}>{link.label}</NavItem>
            ))}
          </nav>

          <button 
            className="md:hidden p-2 -mr-2 text-ink-600 hover:text-ink-900 rounded outline-none focus-visible:ring-2 focus-visible:ring-purple-500"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </header>

      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-cream-50 border-b border-lavender-200 overflow-hidden"
          >
            <nav className="flex flex-col px-4 py-2 gap-2">
              {navLinks.map(link => (
                <NavLink 
                  key={link.to} 
                  to={link.to}
                  onClick={() => setMobileMenuOpen(false)}
                  className={({isActive}) => `px-4 py-3 rounded-md text-sm font-medium ${isActive ? 'bg-lavender-100 text-purple-700' : 'text-ink-600'}`}
                >
                  {link.label}
                </NavLink>
              ))}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="flex-1 w-full max-w-[1120px] mx-auto px-4 md:px-8 py-8 md:py-12">
        <PageTransition>
          <Outlet />
        </PageTransition>
      </main>

      <footer className="mt-auto border-t border-lavender-200 py-8 bg-cream-100/50">
        <div className="max-w-[1120px] mx-auto px-4 md:px-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-sm text-ink-600 flex flex-col md:flex-row md:items-center gap-1 md:gap-4">
            <span className="font-semibold text-ink-900">MargDrishti</span>
            <span className="hidden md:inline text-lavender-300">•</span>
            <span>Academic Data Science Project</span>
          </div>
          
          <div className="flex items-center gap-4 text-xs text-ink-400">
            <span>React • Vite • Tailwind • FastAPI</span>
            <div className="flex items-center gap-1.5 group relative cursor-help">
              <span className={`w-2 h-2 rounded-full ${apiStatus === 'ok' ? 'bg-sage' : apiStatus === 'error' ? 'bg-rose' : 'bg-ink-400'}`}></span>
              <span className="uppercase tracking-wider">API</span>
              
              <div className="absolute bottom-full right-0 mb-2 w-max px-2 py-1 bg-ink-900 text-cream-50 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                {apiStatus === 'ok' ? 'API connected' : apiStatus === 'error' ? 'API disconnected' : 'Checking API...'}
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
""")

write("src/lib/motion.jsx", """
import React, { createContext, useContext, useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { useLocation } from 'react-router-dom';

export const MotionContext = createContext({ reduced: false });

export function MotionProvider({ children }) {
  const reduced = useReducedMotion();
  return <MotionContext.Provider value={{ reduced }}>{children}</MotionContext.Provider>;
}

export function PageTransition({ children }) {
  const location = useLocation();
  const reduced = useReducedMotion();

  if (reduced) return <div key={location.pathname}>{children}</div>;

  return (
    <motion.div
      key={location.pathname}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

export const staggerContainer = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

export const fadeRise = {
  hidden: { opacity: 0, y: 12 },
  show: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] }
  }
};

export function Reveal({ children, className = '', delay = 0 }) {
  const reduced = useReducedMotion();
  
  if (reduced) return <div className={className}>{children}</div>;
  
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

export function CountUp({ to, duration = 1.5, className = '' }) {
  const [count, setCount] = useState(0);
  const reduced = useReducedMotion();

  useEffect(() => {
    if (reduced) {
      setCount(to);
      return;
    }

    let start = null;
    const step = (timestamp) => {
      if (!start) start = timestamp;
      const progress = Math.min((timestamp - start) / (duration * 1000), 1);
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      
      setCount(Math.floor(easeOutQuart * to));
      
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }, [to, duration, reduced]);

  return <span className={className}>{count}</span>;
}
""")

write("src/lib/api.js", """
const TIMEOUT = 10000;

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

/** Get list of available models */
export const getModels = () => fetchWithTimeout('/api/models');

/** Get model details */
export const getModel = (modelId) => fetchWithTimeout(`/api/models/${modelId}`);

/** Get summary metrics for all models */
export const getMetricsSummary = () => fetchWithTimeout('/api/metrics/summary');

/** Get detailed metrics for a model */
export const getMetrics = (modelId) => fetchWithTimeout(`/api/metrics/${modelId}`);

/** Get dataset stats */
export const getDatasetStats = () => fetchWithTimeout('/api/dataset/stats');

/** Get random test samples */
export const getSamples = (limit = 12) => fetchWithTimeout(`/api/samples?limit=${limit}`);

/** Predict a sample */
export const predict = (modelId, imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  return fetchWithTimeout(`/api/predict/${modelId}`, {
    method: 'POST',
    body: formData
  });
};
""")

write("src/hooks/useApi.js", """
import { useState, useEffect, useCallback } from 'react';

const cache = new Map();

export function useApi(apiFunc, ...args) {
  const cacheKey = apiFunc.name + JSON.stringify(args);
  
  const [data, setData] = useState(cache.get(cacheKey) || null);
  const [loading, setLoading] = useState(!cache.has(cacheKey));
  const [error, setError] = useState(null);

  const fetch = useCallback(async (ignoreCache = false) => {
    if (!ignoreCache && cache.has(cacheKey)) {
      setData(cache.get(cacheKey));
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await apiFunc(...args);
      cache.set(cacheKey, result);
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [cacheKey, apiFunc, args]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: () => fetch(true) };
}
""")

write("src/components/ui/index.jsx", """
import React from 'react';
import { motion } from 'motion/react';
import { Link } from 'react-router-dom';

export function Container({ children, className = '' }) {
  return <div className={`w-full max-w-[1120px] mx-auto px-4 md:px-8 ${className}`}>{children}</div>;
}

export function Section({ eyebrow, title, lead, children, className = '' }) {
  return (
    <section className={`py-12 md:py-16 ${className}`}>
      <div className="max-w-3xl mb-12">
        {eyebrow && <div className="text-purple-500 font-semibold tracking-wider uppercase text-sm mb-3">{eyebrow}</div>}
        {title && <h2 className="text-3xl md:text-4xl lg:text-5xl mb-4">{title}</h2>}
        {lead && <p className="text-lg md:text-xl text-ink-600 leading-relaxed">{lead}</p>}
      </div>
      {children}
    </section>
  );
}

export function Card({ children, className = '', hover = false }) {
  return (
    <div className={`bg-white rounded-2xl border border-lavender-200 shadow-soft overflow-hidden transition-all duration-300 ${hover ? 'hover:shadow-lift hover:-translate-y-1' : ''} ${className}`}>
      {children}
    </div>
  );
}

export function StatCard({ label, value, caption, className = '' }) {
  return (
    <Card className={`p-6 ${className}`}>
      <div className="text-sm font-medium text-ink-600 mb-2">{label}</div>
      <div className="text-4xl font-mono text-ink-900 mb-1">{value}</div>
      {caption && <div className="text-xs text-ink-400">{caption}</div>}
    </Card>
  );
}

export const Button = React.forwardRef(({ variant = 'primary', as = 'button', to, href, children, className = '', ...props }, ref) => {
  const baseStyle = "inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2";
  
  const variants = {
    primary: "bg-purple-500 text-white hover:bg-purple-700 shadow-sm",
    secondary: "bg-white border border-lavender-200 text-ink-900 hover:bg-lavender-100 hover:border-lavender-300",
    quiet: "text-purple-700 hover:bg-lavender-100"
  };

  const classes = `${baseStyle} ${variants[variant]} ${className}`;

  if (as === 'Link' && to) {
    return <Link to={to} className={classes} ref={ref} {...props}>{children}</Link>;
  }
  if (as === 'a' && href) {
    return <a href={href} className={classes} ref={ref} {...props}>{children}</a>;
  }
  
  return (
    <button className={classes} ref={ref} {...props}>
      {children}
    </button>
  );
});
Button.displayName = 'Button';

export function EmptyState({ title, description, command, icon: Icon }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-4 text-center rounded-2xl border border-dashed border-lavender-200 bg-cream-100/50">
      {Icon && <Icon size={32} className="text-lavender-300 mb-4" />}
      <h3 className="font-display text-2xl mb-2">{title}</h3>
      <p className="text-ink-600 mb-6 max-w-md">{description}</p>
      {command && (
        <div className="bg-ink-900 text-cream-50 font-mono text-sm px-4 py-3 rounded overflow-x-auto w-full max-w-xl shadow-inner">
          <code className="break-all">{command}</code>
        </div>
      )}
    </div>
  );
}

export function Skeleton({ className = '' }) {
  return <div className={`animate-pulse bg-lavender-100 rounded ${className}`} />;
}
""")

write("src/pages/Home.jsx", """
import React from 'react';
import { motion } from 'motion/react';
import { ArrowRight, ChevronRight, BarChart2 } from 'lucide-react';
import { Reveal, CountUp, staggerContainer, fadeRise } from '../lib/motion';
import { Section, Card, StatCard, Button, EmptyState, Skeleton } from '../components/ui';
import { useApi } from '../hooks/useApi';
import { getMetricsSummary, getDatasetStats, getSamples } from '../lib/api';
import { Link } from 'react-router-dom';

function HeroSamples({ samples, loading }) {
  if (loading) {
    return (
      <div className="grid grid-cols-3 gap-2 p-4 md:p-8 relative h-64 md:h-full">
        <Skeleton className="rounded-xl w-full h-full" />
        <Skeleton className="rounded-xl w-full h-full" />
        <Skeleton className="rounded-xl w-full h-full" />
      </div>
    );
  }

  if (!samples || samples.length === 0) return null;

  return (
    <div className="relative w-full h-64 md:h-96">
      <div className="absolute inset-0 bg-cream-100 rounded-3xl -rotate-2 scale-95 shadow-soft z-0 flex items-center justify-center overflow-hidden">
        <div className="w-64 h-64 rounded-full bg-lavender-100/50 blur-3xl" />
      </div>
      
      <motion.div 
        variants={staggerContainer}
        initial="hidden"
        animate="show"
        className="relative z-10 w-full h-full p-4 md:p-8 grid grid-cols-3 grid-rows-3 gap-2 md:gap-3 lg:gap-4"
      >
        {samples.slice(0, 9).map((sample, i) => (
          <motion.div
            key={i}
            variants={fadeRise}
            whileHover={{ y: -4, scale: 1.02 }}
            className={`rounded-xl md:rounded-2xl overflow-hidden shadow-soft border border-lavender-200 bg-white
              ${i === 4 ? 'col-span-2 row-span-2' : ''}
              ${i === 0 ? 'rounded-tl-3xl' : ''}
              ${i === 2 ? 'rounded-tr-3xl' : ''}
              ${i === 6 ? 'rounded-bl-3xl' : ''}
              ${i === 8 ? 'rounded-br-3xl' : ''}
            `}
          >
            <img src={sample.url} alt={`Traffic sign ${sample.class_id}`} className="w-full h-full object-cover" />
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}

function PipelineFlow() {
  const steps = [
    { label: 'Road images', desc: '4,354 raw Indian scenes' },
    { label: 'Crop signs', desc: 'Bounding box extraction' },
    { label: 'Preprocess', desc: 'Augment & balance' },
    { label: 'Train two models', desc: 'Custom CNN vs ResNet50' },
    { label: 'Compare', desc: 'Accuracy & efficiency' }
  ];

  return (
    <Reveal className="w-full py-12 overflow-x-auto pb-4">
      <div className="flex items-start min-w-[800px]">
        {steps.map((step, i) => (
          <React.Fragment key={i}>
            <div className="flex flex-col items-center flex-1 text-center relative z-10">
              <div className="w-12 h-12 rounded-full bg-cream-50 border-2 border-purple-500 text-purple-700 flex items-center justify-center font-display font-semibold text-xl mb-4 shadow-soft">
                {i + 1}
              </div>
              <h4 className="font-semibold text-ink-900 mb-1">{step.label}</h4>
              <p className="text-xs text-ink-600 px-2">{step.desc}</p>
            </div>
            {i < steps.length - 1 && (
              <div className="flex-1 mt-6 relative h-[2px]">
                <div className="absolute inset-0 bg-lavender-200" />
                <motion.div 
                  className="absolute inset-0 bg-purple-500 origin-left"
                  initial={{ scaleX: 0 }}
                  whileInView={{ scaleX: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8, delay: i * 0.2, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </Reveal>
  );
}

export default function Home() {
  const { data: stats, loading: statsLoading } = useApi(getDatasetStats);
  const { data: metrics, loading: metricsLoading } = useApi(getMetricsSummary);
  const { data: samples, loading: samplesLoading } = useApi(getSamples, 9);

  const margnetAcc = metrics?.models?.['margnet-v5']?.metrics?.accuracy || 0;
  const resnetAcc = metrics?.models?.['resnet50-finetuned']?.metrics?.accuracy || 0;

  const showStats = stats || metrics;

  return (
    <div className="flex flex-col gap-16 md:gap-24">
      {/* Hero */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center min-h-[60vh]">
        <Reveal>
          <div className="text-purple-500 font-semibold tracking-wider uppercase text-sm mb-4">Indian Traffic Sign Recognition</div>
          <h1 className="text-5xl md:text-6xl lg:text-7xl mb-6 leading-[1.1]">Reading the road,<br/>one sign at a time.</h1>
          <p className="text-xl md:text-2xl text-ink-600 leading-relaxed mb-10 max-w-lg">
            How close can a small custom CNN get to a fine-tuned ResNet50 on Indian traffic signs? 
            And at what fraction of the cost?
          </p>
          <div className="flex flex-wrap gap-4">
            <Button as="Link" to="/demo" variant="primary">
              Try the live demo <ChevronRight size={18} />
            </Button>
            <Button as="Link" to="/comparison" variant="secondary">
              See the comparison
            </Button>
          </div>
        </Reveal>
        <Reveal delay={0.2} className="relative w-full">
          <HeroSamples samples={samples} loading={samplesLoading} />
        </Reveal>
      </section>

      {/* Stats */}
      {showStats ? (
        <Reveal>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
            <StatCard 
              label="Classes" 
              value={statsLoading ? <Skeleton className="w-16 h-10" /> : <CountUp to={stats?.stats?.num_classes || 0} />} 
              caption="Unique Indian traffic signs" 
            />
            <StatCard 
              label="Cropped Signs" 
              value={statsLoading ? <Skeleton className="w-24 h-10" /> : <CountUp to={stats?.stats?.total_crops || 0} />} 
              caption="From 4,354 source images" 
            />
            <StatCard 
              label="MargNet Accuracy" 
              value={metricsLoading ? <Skeleton className="w-24 h-10" /> : <><CountUp to={Math.round(margnetAcc * 100)} />%</>} 
              caption="Test set macro F1" 
              className="border-purple-200 bg-purple-50/50"
            />
            <StatCard 
              label="ResNet50 Accuracy" 
              value={metricsLoading ? <Skeleton className="w-24 h-10" /> : <><CountUp to={Math.round(resnetAcc * 100)} />%</>} 
              caption="Test set macro F1" 
              className="border-ochre/20 bg-ochre/5"
            />
          </div>
        </Reveal>
      ) : (
        <EmptyState 
          icon={BarChart2}
          title="No data available yet" 
          description="Run the data processing and training scripts to generate the models and stats."
          command="python -m scripts.run_all_reports"
        />
      )}

      {/* The Question */}
      <Reveal className="max-w-4xl mx-auto text-center py-8">
        <blockquote className="font-display text-3xl md:text-4xl text-ink-900 leading-snug mb-6">
          "Can a model with less than 2% of ResNet50's parameters accurately classify Indian traffic signs on a CPU in real-time?"
        </blockquote>
        <p className="text-ink-600 text-lg">
          This project explores the trade-offs between model size, inference speed, and accuracy 
          in the context of autonomous driving assistance systems in developing regions.
        </p>
      </Reveal>

      {/* Pipeline */}
      <Section title="The Pipeline" className="bg-cream-100/30 rounded-3xl p-8 border border-lavender-200">
        <PipelineFlow />
      </Section>

      {/* Models */}
      <Section title="The Contenders" lead="Two very different architectures trained on the exact same dataset splits.">
        {metricsLoading ? (
          <div className="grid md:grid-cols-2 gap-8">
            <Skeleton className="h-64 rounded-2xl" />
            <Skeleton className="h-64 rounded-2xl" />
          </div>
        ) : metrics?.models ? (
          <div className="grid md:grid-cols-2 gap-8">
            {/* MargNet */}
            <Card className="p-8 border-t-4 border-t-purple-500">
              <h3 className="text-2xl font-display font-semibold mb-2">MargNet</h3>
              <p className="text-ink-600 mb-6 h-12">A lightweight custom CNN designed specifically for road sign classification.</p>
              
              <div className="space-y-4 mb-8">
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Parameters</span>
                  <span className="font-mono">{metrics.models['margnet-v5']?.metrics?.params?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Size</span>
                  <span className="font-mono">{metrics.models['margnet-v5']?.metrics?.model_size_mb?.toFixed(1) || 'N/A'} MB</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Inference (CPU)</span>
                  <span className="font-mono">{metrics.models['margnet-v5']?.metrics?.latency_cpu_ms?.toFixed(1) || 'N/A'} ms</span>
                </div>
              </div>
              
              <Button as="Link" to="/models" variant="quiet" className="w-full -ml-4">
                View model details <ArrowRight size={16} />
              </Button>
            </Card>

            {/* ResNet50 */}
            <Card className="p-8 border-t-4 border-t-ochre">
              <h3 className="text-2xl font-display font-semibold mb-2">ResNet50</h3>
              <p className="text-ink-600 mb-6 h-12">The industry standard, fine-tuned from ImageNet weights.</p>
              
              <div className="space-y-4 mb-8">
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Parameters</span>
                  <span className="font-mono">{metrics.models['resnet50-finetuned']?.metrics?.params?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Size</span>
                  <span className="font-mono">{metrics.models['resnet50-finetuned']?.metrics?.model_size_mb?.toFixed(1) || 'N/A'} MB</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Inference (CPU)</span>
                  <span className="font-mono">{metrics.models['resnet50-finetuned']?.metrics?.latency_cpu_ms?.toFixed(1) || 'N/A'} ms</span>
                </div>
              </div>
              
              <Button as="Link" to="/models" variant="quiet" className="w-full -ml-4 !text-ochre hover:!bg-ochre/10">
                View model details <ArrowRight size={16} />
              </Button>
            </Card>
          </div>
        ) : (
           <EmptyState 
            icon={BarChart2}
            title="Models not evaluated" 
            description="Run the evaluation scripts to see model details."
            command="python -m scripts.run_all_reports"
          />
        )}
      </Section>

      {/* CTA */}
      <Reveal className="bg-lavender-100 rounded-3xl p-10 md:p-16 text-center border border-lavender-200 shadow-inner">
        <h2 className="text-3xl md:text-4xl mb-4">Try it yourself</h2>
        <p className="text-lg text-ink-600 mb-8 max-w-2xl mx-auto">
          Upload an image of an Indian traffic sign or use one from our test set to see how both models perform in real-time.
        </p>
        <Button as="Link" to="/demo" variant="primary" className="text-base px-8 py-4">
          Open Live Demo
        </Button>
      </Reveal>
    </div>
  );
}
""")

write("src/pages/Dataset.jsx", """
import React from 'react';
import { Section, EmptyState } from '../components/ui';
import { BarChart2 } from 'lucide-react';

export default function Dataset() {
  return (
    <div>
      <Section title="Dataset" lead="Indian Traffic SignBoards from Roboflow Universe." />
      <EmptyState 
        icon={BarChart2}
        title="Dataset view not implemented" 
        description="Run dataset reports to populate this section."
      />
    </div>
  );
}
""")

write("src/pages/Models.jsx", """
import React from 'react';
import { Section, EmptyState } from '../components/ui';
import { BarChart2 } from 'lucide-react';

export default function Models() {
  return (
    <div>
      <Section title="Models" lead="Architecture details and training history." />
      <EmptyState 
        icon={BarChart2}
        title="Models view not implemented" 
        description="Run training and evaluation scripts first."
      />
    </div>
  );
}
""")

write("src/pages/LiveDemo.jsx", """
import React from 'react';
import { Section, EmptyState } from '../components/ui';
import { BarChart2 } from 'lucide-react';

export default function LiveDemo() {
  return (
    <div>
      <Section title="Live Demo" lead="Upload an image to test the models." />
      <EmptyState 
        icon={BarChart2}
        title="Demo not implemented" 
        description="The backend inference API is ready, UI coming soon."
      />
    </div>
  );
}
""")

write("src/pages/Comparison.jsx", """
import React from 'react';
import { Section, EmptyState } from '../components/ui';
import { BarChart2 } from 'lucide-react';

export default function Comparison() {
  return (
    <div>
      <Section title="Comparison" lead="Detailed performance and efficiency metrics." />
      <EmptyState 
        icon={BarChart2}
        title="Comparison not implemented" 
        description="Run benchmark scripts to generate comparison data."
      />
    </div>
  );
}
""")

write("src/pages/About.jsx", """
import React from 'react';
import { Section } from '../components/ui';

export default function About() {
  return (
    <div>
      <Section title="About" lead="Project background and methodology." />
      <div className="prose prose-purple max-w-3xl text-ink-600">
        <p>A college Data Science lab project exploring CNN architectures for Indian traffic signs.</p>
      </div>
    </div>
  );
}
""")

write("src/pages/NotFound.jsx", """
import React from 'react';
import { EmptyState, Button } from '../components/ui';
import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="min-h-[50vh] flex flex-col items-center justify-center">
      <h1 className="text-6xl font-display mb-4 text-purple-700">404</h1>
      <p className="text-xl text-ink-600 mb-8">Page not found</p>
      <Button as="Link" to="/">Return Home</Button>
    </div>
  );
}
""")

print("Frontend scaffold generated.")
