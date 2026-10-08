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
