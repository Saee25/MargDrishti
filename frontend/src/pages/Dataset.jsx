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
