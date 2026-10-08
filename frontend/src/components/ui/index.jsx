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
