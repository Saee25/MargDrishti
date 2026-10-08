import React, { useState, useMemo } from 'react';
import { Search } from 'lucide-react';
import { Section, Card, StatCard, Container, Skeleton } from '../components/ui';
import { BarChartV, BarChartH, Histogram } from '../components/charts';
import { Reveal, staggerContainer, fadeRise } from '../lib/motion';
import { useApi } from '../hooks/useApi';
import { getDatasetStats, getSamples } from '../lib/api';

export default function Dataset() {
  const { data: stats, loading } = useApi(getDatasetStats);
  const { data: samples } = useApi(getSamples, 12);
  const [searchTerm, setSearchTerm] = useState('');
  const [splitView, setSplitView] = useState('train');

  if (loading || !stats) {
    return <Container className="py-12"><Skeleton className="h-96" /></Container>;
  }

  const { stats: s, class_stats, dropped_classes } = stats;

  const splitData = [
    { name: 'Train', count: s.train_crops },
    { name: 'Validation', count: s.valid_crops },
    { name: 'Test', count: s.test_crops }
  ];

  const filteredClassStats = useMemo(() => {
    return (class_stats || [])
      .filter(c => c.name.toLowerCase().includes(searchTerm.toLowerCase()))
      .sort((a, b) => b[splitView] - a[splitView])
      .map(c => ({ name: c.name, count: c[splitView] }));
  }, [class_stats, searchTerm, splitView]);

  return (
    <div className="flex flex-col gap-12">
      <Section title="The Dataset" lead="From raw road images to cropped traffic signs.">
        <div className="grid md:grid-cols-5 gap-6 mb-12">
          <StatCard label="Source Images" value={s.total_images} caption="Roboflow Universe" />
          <StatCard label="Total Crops" value={s.total_crops} caption="Valid bounding boxes" />
          <StatCard label="Classes Kept" value={s.num_classes} caption="≥ 40 crops each" />
          <StatCard label="Classes Dropped" value={s.dropped_classes} caption="Too few samples" />
          <StatCard label="Imbalance" value={`${Math.round(s.max_class_crops / (s.min_class_crops || 1))}x`} caption="Max to min ratio" />
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          <Card className="p-6">
            <BarChartV data={splitData} keys={['count']} colors={['#9479B8']} title="Dataset Splits" caption="Crop counts by split" summary="Bar chart of dataset splits" />
          </Card>
          <Card className="p-6">
            <h4 className="font-semibold mb-4 text-ink-900">Crop Pipeline</h4>
            <div className="flex items-center gap-4 text-sm text-ink-600">
              <div className="flex-1 border border-lavender-200 p-2 rounded text-center">Original Image<br/>+ Box</div>
              <div className="text-lavender-300">→</div>
              <div className="flex-1 border border-lavender-200 p-2 rounded text-center">Crop<br/>(10% padding)</div>
              <div className="text-lavender-300">→</div>
              <div className="flex-1 border border-lavender-200 p-2 rounded text-center">Resize<br/>64px & 224px</div>
            </div>
            <p className="mt-4 text-xs text-ink-600">Bounding boxes whose shorter side is under 20px are skipped.</p>
          </Card>
        </div>

        <Card className="p-6 mb-12">
          <div className="flex justify-between items-end mb-6 flex-wrap gap-4">
            <div>
              <h4 className="font-semibold text-ink-900 mb-1">Class Distribution</h4>
              <p className="text-sm text-ink-600">Search and explore classes</p>
            </div>
            <div className="flex gap-4">
              <div className="relative">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
                <input 
                  type="text" 
                  placeholder="Search classes..." 
                  className="pl-9 pr-4 py-1.5 border border-lavender-200 rounded-lg text-sm bg-cream-50 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                />
              </div>
              <div className="flex bg-lavender-100 p-1 rounded-lg">
                {['train', 'valid', 'test'].map(s => (
                  <button 
                    key={s}
                    onClick={() => setSplitView(s)}
                    className={`px-3 py-1 text-xs font-medium rounded-md capitalize ${splitView === s ? 'bg-white shadow-sm text-purple-700' : 'text-ink-600'}`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="h-[400px] overflow-y-auto pr-2 custom-scrollbar">
            <BarChartH data={filteredClassStats} keys={['count']} colors={['#7C5FA6']} title="" caption="" summary="Horizontal bar chart of classes" />
          </div>
        </Card>

        {dropped_classes && dropped_classes.length > 0 && (
          <Card className="p-6 mb-12">
            <h4 className="font-semibold mb-4 text-ink-900">Dropped Classes</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-ink-600 uppercase bg-lavender-100/50">
                  <tr>
                    <th className="px-4 py-3 rounded-tl-lg">Class Name</th>
                    <th className="px-4 py-3 rounded-tr-lg text-right">Crops</th>
                  </tr>
                </thead>
                <tbody>
                  {dropped_classes.map((c, i) => (
                    <tr key={i} className="border-b border-lavender-100 last:border-0">
                      <td className="px-4 py-3 font-medium text-ink-900">{c.name}</td>
                      <td className="px-4 py-3 text-right font-mono text-ink-600">{c.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </Section>
    </div>
  );
}
