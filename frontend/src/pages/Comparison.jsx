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
