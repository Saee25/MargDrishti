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
