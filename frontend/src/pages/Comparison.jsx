import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { toPng } from 'html-to-image';
import { Download, Copy, Presentation, ArrowRight, ArrowLeft } from 'lucide-react';
import { formatNumber, formatPercent, formatParams, formatSize, formatTime } from '../lib/format';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis, LineChart, Line, ComposedChart } from 'recharts';

function ChartExportBtn({ chartRef, filename }) {
  const handleDownload = () => {
    if (chartRef.current === null) return;
    toPng(chartRef.current, { cacheBust: true, backgroundColor: '#FFFDF3' })
      .then((dataUrl) => {
        const link = document.createElement('a');
        link.download = filename;
        link.href = dataUrl;
        link.click();
      })
      .catch((err) => {
        console.error('Oops, something went wrong!', err);
      });
  };

  return (
    <button 
      onClick={handleDownload}
      className="absolute top-2 right-2 p-2 bg-white/80 hover:bg-white rounded-lg border border-lavender-200 text-ink-600 hover:text-purple-700 transition-colors shadow-sm z-10"
      title="Download PNG"
    >
      <Download size={16} />
    </button>
  );
}

export default function Comparison() {
  const [data, setData] = useState(null);
  const [historyMargnet, setHistoryMargnet] = useState([]);
  const [historyResnet, setHistoryResnet] = useState([]);
  const [ablationData, setAblationData] = useState([]);
  
  const [presentationMode, setPresentationMode] = useState(false);
  const [currentSection, setCurrentSection] = useState(0);
  const [showFrozen, setShowFrozen] = useState(true);

  const sections = [
    { id: 'verdict', title: 'The Verdict' },
    { id: 'headline', title: 'Headline Metrics' },
    { id: 'accuracy', title: 'Accuracy & F1' },
    { id: 'efficiency', title: 'Efficiency' },
    { id: 'training', title: 'Training Behaviour' },
    { id: 'mistakes', title: 'Mistakes' },
    { id: 'class', title: 'Class by Class' },
    { id: 'confusion', title: 'Confusion Matrix' },
    { id: 'robustness', title: 'Robustness' },
    { id: 'gradcam', title: 'Grad-CAM' },
    { id: 'ablation', title: 'Ablation Recap' },
    { id: 'limitations', title: 'Limitations' }
  ];

  const sectionRefs = useRef(sections.map(() => React.createRef()));

  useEffect(() => {
    fetch('/api/metrics/summary')
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
      
    fetch('/api/metrics/margnet/history')
      .then(res => res.json())
      .then(d => setHistoryMargnet(d.history))
      .catch(console.error);

    fetch('/api/metrics/resnet50/history')
      .then(res => res.json())
      .then(d => setHistoryResnet(d.history))
      .catch(console.error);
      
    fetch('/api/metrics/ablation')
      .then(res => res.json())
      .then(d => setAblationData(d.rows))
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!presentationMode) {
      const handleScroll = () => {
        const scrollPosition = window.scrollY + 100;
        const current = sectionRefs.current.findIndex((ref) => {
          if (!ref.current) return false;
          const top = ref.current.offsetTop;
          const height = ref.current.offsetHeight;
          return scrollPosition >= top && scrollPosition < top + height;
        });
        if (current !== -1 && current !== currentSection) {
          setCurrentSection(current);
        }
      };
      window.addEventListener('scroll', handleScroll);
      return () => window.removeEventListener('scroll', handleScroll);
    }
  }, [presentationMode, currentSection]);

  useEffect(() => {
    if (presentationMode) {
      const handleKeyDown = (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
          setCurrentSection(prev => Math.min(prev + 1, sections.length - 1));
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
          setCurrentSection(prev => Math.max(prev - 1, 0));
        }
      };
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [presentationMode, sections.length]);

  useEffect(() => {
    if (presentationMode && sectionRefs.current[currentSection]?.current) {
      sectionRefs.current[currentSection].current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentSection, presentationMode]);

  if (!data) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  const margnetRow = data.rows?.find(m => m.display_name === 'MargNet') || {};
  const resnetFineRow = data.rows?.find(m => m.display_name === 'ResNet50 Fine-tuned') || {};
  const resnetFrozenRow = data.rows?.find(m => m.display_name === 'ResNet50 Frozen') || {};

  const accGap = ((resnetFineRow.macro_f1 || 0) - (margnetRow.macro_f1 || 0)) * 100;
  const paramRatio = ((resnetFineRow.parameters || 1) / (margnetRow.parameters || 1)).toFixed(1);
  const sizeRatio = ((resnetFineRow.size_mb || 1) / (margnetRow.size_mb || 1)).toFixed(1);
  const speedRatio = ((resnetFineRow.cpu_latency_ms || 1) / (margnetRow.cpu_latency_ms || 1)).toFixed(1);

  const accChartData = [
    { name: 'Accuracy', MargNet: margnetRow.test_accuracy || 0, ResNet50: resnetFineRow.test_accuracy || 0 },
    { name: 'Top-3 Acc', MargNet: margnetRow.top3_accuracy || 0, ResNet50: resnetFineRow.top3_accuracy || 0 },
    { name: 'Macro F1', MargNet: margnetRow.macro_f1 || 0, ResNet50: resnetFineRow.macro_f1 || 0 },
  ].map(d => ({ ...d, MargNet: d.MargNet * 100, ResNet50: d.ResNet50 * 100 }));
  
  const effChartData = [
    { name: 'Parameters (M)', MargNet: (margnetRow.parameters || 0) / 1e6, ResNet50: (resnetFineRow.parameters || 0) / 1e6 },
    { name: 'Size (MB)', MargNet: margnetRow.size_mb || 0, ResNet50: resnetFineRow.size_mb || 0 },
    { name: 'Latency (ms)', MargNet: margnetRow.cpu_latency_ms || 0, ResNet50: resnetFineRow.cpu_latency_ms || 0 },
  ];

  const accChartRef = React.createRef();
  const effChartRef = React.createRef();
  const trainChartRef = React.createRef();
  const ablChartRef = React.createRef();

  return (
    <div className={`flex gap-8 relative ${presentationMode ? 'text-xl' : ''}`}>
      {/* Side Navigation */}
      {!presentationMode && (
        <div className="w-64 shrink-0 hidden lg:block relative">
          <div className="sticky top-24 space-y-2">
            <h3 className="font-medium text-ink-900 mb-4 px-3">Contents</h3>
            <div className="space-y-1">
              {sections.map((s, i) => (
                <button
                  key={s.id}
                  onClick={() => {
                    setCurrentSection(i);
                    sectionRefs.current[i].current?.scrollIntoView({ behavior: 'smooth' });
                  }}
                  className={`block w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                    currentSection === i 
                      ? 'bg-purple-100 text-purple-700 font-medium' 
                      : 'text-ink-600 hover:bg-lavender-100 hover:text-ink-900'
                  }`}
                >
                  {i + 1}. {s.title}
                </button>
              ))}
            </div>
            <button
              onClick={() => setPresentationMode(true)}
              className="mt-8 flex items-center gap-2 w-full px-3 py-2 text-sm text-purple-700 bg-purple-50 hover:bg-purple-100 rounded-lg transition-colors border border-purple-200"
            >
              <Presentation size={16} />
              Presentation Mode
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className={`flex-1 space-y-24 pb-32 ${presentationMode ? 'max-w-5xl mx-auto' : 'max-w-4xl'}`}>

        {presentationMode && (
          <div className="fixed top-4 right-4 z-50 flex gap-4">
            <button
              onClick={() => setPresentationMode(false)}
              className="px-4 py-2 bg-white rounded-lg shadow-md border border-lavender-200 text-ink-900 hover:bg-lavender-100"
            >
              Exit Presentation
            </button>
          </div>
        )}

        {/* 1. Verdict */}
        <section ref={sectionRefs.current[0]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">1. The Verdict</span>
            <h2 className="text-3xl font-serif text-ink-900">How close can a custom CNN get?</h2>
          </div>
          <p className="text-lg text-ink-600 leading-relaxed">
            MargNet achieves a macro F1 score within {accGap.toFixed(1)} percentage points of a fine-tuned ResNet50. 
            However, it does so using {paramRatio}x fewer parameters, is {sizeRatio}x smaller on disk, and runs {speedRatio}x faster on a standard CPU.
          </p>
          <div className="grid grid-cols-3 gap-6">
            <div className="bg-white p-6 rounded-2xl border border-lavender-200 text-center">
              <div className="text-3xl font-mono text-purple-700 mb-1">{paramRatio}x</div>
              <div className="text-sm text-ink-600">fewer parameters</div>
            </div>
            <div className="bg-white p-6 rounded-2xl border border-lavender-200 text-center">
              <div className="text-3xl font-mono text-purple-700 mb-1">{sizeRatio}x</div>
              <div className="text-sm text-ink-600">smaller size</div>
            </div>
            <div className="bg-white p-6 rounded-2xl border border-lavender-200 text-center">
              <div className="text-3xl font-mono text-purple-700 mb-1">{speedRatio}x</div>
              <div className="text-sm text-ink-600">faster inference</div>
            </div>
          </div>
        </section>

        {/* 2. Headline Metrics */}
        <section ref={sectionRefs.current[1]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">2. Headline Metrics</span>
            <h2 className="text-3xl font-serif text-ink-900">The Bottom Line</h2>
          </div>
          <div className="flex justify-between items-center">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={showFrozen} onChange={(e) => setShowFrozen(e.target.checked)} className="rounded text-purple-600" />
              <span className="text-sm text-ink-600">Show ResNet50 frozen</span>
            </label>
            <button className="flex items-center gap-2 px-3 py-1.5 text-sm bg-white border border-lavender-200 rounded-lg hover:bg-lavender-50 transition-colors text-ink-600">
              <Copy size={14} /> Copy as table
            </button>
          </div>
          <div className="overflow-x-auto bg-white rounded-2xl border border-lavender-200 shadow-sm">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-lavender-50 border-b border-lavender-200">
                <tr>
                  <th className="px-6 py-4 font-medium text-ink-900">Metric</th>
                  <th className="px-6 py-4 font-medium text-ink-900">MargNet</th>
                  {showFrozen && <th className="px-6 py-4 font-medium text-ink-900">ResNet50 frozen</th>}
                  <th className="px-6 py-4 font-medium text-ink-900">ResNet50 fine-tuned</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-lavender-100">
                <tr>
                  <td className="px-6 py-4 text-ink-600">Accuracy</td>
                  <td className="px-6 py-4 font-mono">{formatPercent(margnetRow.test_accuracy)}</td>
                  {showFrozen && <td className="px-6 py-4 font-mono text-ink-400">{formatPercent(resnetFrozenRow.test_accuracy)}</td>}
                  <td className="px-6 py-4 font-mono bg-sage/10 text-sage-700 font-medium">{formatPercent(resnetFineRow.test_accuracy)}</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 text-ink-600">Macro F1</td>
                  <td className="px-6 py-4 font-mono">{formatPercent(margnetRow.macro_f1)}</td>
                  {showFrozen && <td className="px-6 py-4 font-mono text-ink-400">{formatPercent(resnetFrozenRow.macro_f1)}</td>}
                  <td className="px-6 py-4 font-mono bg-sage/10 text-sage-700 font-medium">{formatPercent(resnetFineRow.macro_f1)}</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 text-ink-600">Parameters</td>
                  <td className="px-6 py-4 font-mono bg-sage/10 text-sage-700 font-medium">{formatParams(margnetRow.parameters)}</td>
                  {showFrozen && <td className="px-6 py-4 font-mono text-ink-400">{formatParams(resnetFrozenRow.parameters)}</td>}
                  <td className="px-6 py-4 font-mono">{formatParams(resnetFineRow.parameters)}</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 text-ink-600">Size</td>
                  <td className="px-6 py-4 font-mono bg-sage/10 text-sage-700 font-medium">{formatSize(margnetRow.size_mb)}</td>
                  {showFrozen && <td className="px-6 py-4 font-mono text-ink-400">{formatSize(resnetFrozenRow.size_mb)}</td>}
                  <td className="px-6 py-4 font-mono">{formatSize(resnetFineRow.size_mb)}</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 text-ink-600">CPU Latency</td>
                  <td className="px-6 py-4 font-mono bg-sage/10 text-sage-700 font-medium">{formatTime(margnetRow.cpu_latency_ms)}</td>
                  {showFrozen && <td className="px-6 py-4 font-mono text-ink-400">{formatTime(resnetFrozenRow.cpu_latency_ms)}</td>}
                  <td className="px-6 py-4 font-mono">{formatTime(resnetFineRow.cpu_latency_ms)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 3. Accuracy & F1 */}
        <section ref={sectionRefs.current[2]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">3. Accuracy & F1</span>
            <h2 className="text-3xl font-serif text-ink-900">Performance across metrics</h2>
          </div>
          <div className="h-80 relative bg-white p-6 rounded-2xl border border-lavender-200" ref={accChartRef}>
            <ChartExportBtn chartRef={accChartRef} filename="accuracy_metrics.png" />
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={accChartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EEEDFB" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} />
                <YAxis domain={[80, 100]} axisLine={false} tickLine={false} tickFormatter={(val) => `${val}%`} />
                <Tooltip formatter={(val) => `${val.toFixed(1)}%`} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Legend iconType="circle" />
                <Bar dataKey="MargNet" fill="#7C5FA6" radius={[4, 4, 0, 0]} maxBarSize={40} />
                <Bar dataKey="ResNet50" fill="#C39A45" radius={[4, 4, 0, 0]} maxBarSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* 4. Efficiency */}
        <section ref={sectionRefs.current[3]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">4. Efficiency</span>
            <h2 className="text-3xl font-serif text-ink-900">Size and Speed Tradeoffs</h2>
          </div>
          <div className="h-80 relative bg-white p-6 rounded-2xl border border-lavender-200" ref={effChartRef}>
            <ChartExportBtn chartRef={effChartRef} filename="efficiency.png" />
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={effChartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#EEEDFB" />
                <XAxis type="number" axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" axisLine={false} tickLine={false} width={120} />
                <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Legend iconType="circle" />
                <Bar dataKey="MargNet" fill="#7C5FA6" radius={[0, 4, 4, 0]} maxBarSize={30} />
                <Bar dataKey="ResNet50" fill="#C39A45" radius={[0, 4, 4, 0]} maxBarSize={30} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
        
        {/* 5. Training Behaviour */}
        <section ref={sectionRefs.current[4]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">5. Training Behaviour</span>
            <h2 className="text-3xl font-serif text-ink-900">Convergence over Epochs</h2>
          </div>
          <div className="h-96 relative bg-white p-6 rounded-2xl border border-lavender-200" ref={trainChartRef}>
            <ChartExportBtn chartRef={trainChartRef} filename="training_behaviour.png" />
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EEEDFB" />
                <XAxis dataKey="epoch" axisLine={false} tickLine={false} type="number" domain={['dataMin', 'dataMax']} allowDuplicatedCategory={false} />
                <YAxis yAxisId="left" domain={[0, 'auto']} axisLine={false} tickLine={false} label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 1]} axisLine={false} tickLine={false} tickFormatter={(v) => `${(v*100).toFixed(0)}%`} label={{ value: 'Accuracy', angle: 90, position: 'insideRight' }} />
                <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Legend iconType="circle" />
                {historyMargnet && <Line yAxisId="left" type="monotone" data={historyMargnet} dataKey="val_loss" name="MargNet Val Loss" stroke="#7C5FA6" strokeWidth={2} dot={false} />}
                {historyResnet && <Line yAxisId="left" type="monotone" data={historyResnet} dataKey="val_loss" name="ResNet50 Val Loss" stroke="#C39A45" strokeWidth={2} dot={false} />}
                {historyMargnet && <Line yAxisId="right" type="monotone" data={historyMargnet} dataKey="val_acc" name="MargNet Val Acc" stroke="#A995C9" strokeWidth={2} strokeDasharray="5 5" dot={false} />}
                {historyResnet && <Line yAxisId="right" type="monotone" data={historyResnet} dataKey="val_acc" name="ResNet50 Val Acc" stroke="#D8BD7C" strokeWidth={2} strokeDasharray="5 5" dot={false} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* 6. Mistakes */}
        <section ref={sectionRefs.current[5]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">6. Mistakes</span>
            <h2 className="text-3xl font-serif text-ink-900">Where Models Go Wrong</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white rounded-2xl border border-lavender-200 overflow-hidden">
                <div className="px-4 py-2 bg-lavender-50 border-b border-lavender-200 text-sm font-medium">MargNet Misclassifications</div>
                <img src="/static/figures/margnet_misclassified_grid.png" alt="MargNet Mistakes" className="w-full h-auto object-cover p-2" />
            </div>
            <div className="bg-white rounded-2xl border border-lavender-200 overflow-hidden">
                <div className="px-4 py-2 bg-lavender-50 border-b border-lavender-200 text-sm font-medium">ResNet50 Misclassifications</div>
                <img src="/static/figures/resnet50_misclassified_grid.png" alt="ResNet50 Mistakes" className="w-full h-auto object-cover p-2" />
            </div>
          </div>
        </section>

        {/* 7. Class by Class */}
        <section ref={sectionRefs.current[6]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">7. Class by Class</span>
            <h2 className="text-3xl font-serif text-ink-900">Per-Class F1 Score Difference</h2>
          </div>
          <div className="bg-white rounded-2xl border border-lavender-200 overflow-hidden">
             <img src="/static/figures/per_class_f1_difference.png" alt="Per Class Difference" className="w-full h-auto" />
          </div>
        </section>

        {/* 8. Confusion Matrix */}
        <section ref={sectionRefs.current[7]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">8. Confusion Matrix</span>
            <h2 className="text-3xl font-serif text-ink-900">Side-by-Side Comparison</h2>
          </div>
          <div className="bg-white rounded-2xl border border-lavender-200 overflow-hidden p-2">
             <img src="/static/figures/confusion_side_by_side.png" alt="Confusion Matrix Comparison" className="w-full h-auto" />
          </div>
        </section>

        {/* 9. Robustness */}
        <section ref={sectionRefs.current[8]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">9. Robustness</span>
            <h2 className="text-3xl font-serif text-ink-900">Accuracy by Crop Size</h2>
          </div>
          <div className="bg-white rounded-2xl border border-lavender-200 overflow-hidden p-2">
             <img src="/static/figures/accuracy_by_crop_size.png" alt="Accuracy by Crop Size" className="w-full h-auto max-w-2xl mx-auto" />
          </div>
        </section>

        {/* 10. Grad-CAM */}
        <section ref={sectionRefs.current[9]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">10. Grad-CAM</span>
            <h2 className="text-3xl font-serif text-ink-900">Visualizing Attention</h2>
          </div>
          <div className="bg-white rounded-2xl border border-lavender-200 overflow-hidden p-2">
             <img src="/static/figures/gradcam_comparison.png" alt="Grad-CAM Analysis" className="w-full h-auto" />
          </div>
        </section>

        {/* 11. Ablation Recap */}
        <section ref={sectionRefs.current[10]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">11. Ablation Recap</span>
            <h2 className="text-3xl font-serif text-ink-900">Ablation Study (MargNet Evolution)</h2>
          </div>
          <div className="h-80 relative bg-white p-6 rounded-2xl border border-lavender-200" ref={ablChartRef}>
            <ChartExportBtn chartRef={ablChartRef} filename="ablation_study.png" />
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={ablationData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EEEDFB" />
                <XAxis dataKey="step" axisLine={false} tickLine={false} />
                <YAxis yAxisId="left" domain={[60, 100]} axisLine={false} tickLine={false} tickFormatter={(val) => `${val}%`} />
                <YAxis yAxisId="right" orientation="right" axisLine={false} tickLine={false} tickFormatter={(val) => formatParams(val)} />
                <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Legend iconType="circle" />
                <Bar yAxisId="left" dataKey="test_acc" name="Test Accuracy (%)" fill="#7C5FA6" radius={[4, 4, 0, 0]} maxBarSize={40} />
                <Line yAxisId="right" type="monotone" dataKey="parameters" name="Parameters" stroke="#C39A45" strokeWidth={3} dot={{ r: 4 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* 12. Limitations */}
        <section ref={sectionRefs.current[11]} className="space-y-6 scroll-mt-24">
          <div className="space-y-2">
            <span className="text-sm font-medium text-purple-500 uppercase tracking-wider">12. Limitations</span>
            <h2 className="text-3xl font-serif text-ink-900">Honest Notes</h2>
          </div>
          <div className="bg-lavender-50 p-6 rounded-2xl border border-lavender-200">
            <ul className="space-y-3 text-ink-600 list-disc list-inside">
              <li>Small dataset with many classes having very few images, leading to noisy per-class metrics.</li>
              <li>Classification is performed on cropped signs only, not full images.</li>
              <li>Source images were pre-stretched to 640x640 before cropping, distorting aspect ratios.</li>
              <li>ResNet50 was trained on images resized to 224x224, losing some high-frequency detail.</li>
              <li>Training times were measured on a Google Colab T4 GPU, while CPU inference was measured locally.</li>
            </ul>
          </div>
        </section>

      </div>
    </div>
  );
}
