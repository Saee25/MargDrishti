import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './app/Layout';
import Home from './pages/Home';
import Dataset from './pages/Dataset';
import Models from './pages/Models';
import LiveDemo from './pages/LiveDemo';
import Comparison from './pages/Comparison';
import About from './pages/About';
import NotFound from './pages/NotFound';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="dataset" element={<Dataset />} />
          <Route path="models" element={<Models />} />
          <Route path="demo" element={<LiveDemo />} />
          <Route path="comparison" element={<Comparison />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
