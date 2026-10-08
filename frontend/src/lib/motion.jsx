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
