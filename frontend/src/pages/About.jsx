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
