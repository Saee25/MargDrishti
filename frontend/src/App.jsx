import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './app/Layout';
import Home from './pages/Home';
import Dataset from './pages/Dataset';
import Models from './pages/Models';
import LiveDemo from './pages/LiveDemo';
import About from './pages/About';
import NotFound from './pages/NotFound';

const Comparison = lazy(() => import('./pages/Comparison'));

function Loading() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="w-8 h-8 rounded-full border-4 border-lavender-200 border-t-purple-500 animate-spin"></div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="dataset" element={<Dataset />} />
          <Route path="models" element={<Models />} />
          <Route path="demo" element={<LiveDemo />} />
          <Route path="comparison" element={
            <Suspense fallback={<Loading />}>
              <Comparison />
            </Suspense>
          } />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
