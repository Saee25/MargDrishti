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
