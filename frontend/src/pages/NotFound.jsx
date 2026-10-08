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
