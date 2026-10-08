import React, { useState } from 'react';
import { Section, Card, Container } from '../components/ui';
import { Reveal } from '../lib/motion';

function ArchitectureBlock({ name, details, shape }) {
  return (
    <div className="flex flex-col items-center group relative cursor-help">
      <div className="bg-white border border-lavender-200 rounded-lg p-3 text-center shadow-sm group-hover:border-purple-500 group-hover:shadow-md transition-all">
        <span className="text-sm font-semibold text-purple-700 block mb-1">{name}</span>
        <span className="text-xs text-ink-600">{details}</span>
      </div>
      <div className="mt-2 text-[10px] font-mono text-ink-400">{shape}</div>
    </div>
  );
}

export default function Models() {
  const [tab, setTab] = useState('margnet');

  return (
    <div className="flex flex-col gap-12">
      <Section title="Models" lead="Two approaches to the same problem.">
        <div className="flex bg-lavender-100 p-1 rounded-xl w-fit mb-8">
          {[
            { id: 'margnet', label: 'MargNet' },
            { id: 'resnet', label: 'ResNet50' },
            { id: 'sidebyside', label: 'Side by Side' }
          ].map(t => (
            <button 
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-6 py-2 text-sm font-medium rounded-lg transition-colors ${tab === t.id ? 'bg-white shadow-sm text-purple-700' : 'text-ink-600 hover:text-ink-900'}`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === 'margnet' && (
          <Reveal>
            <Card className="p-8 bg-cream-50">
              <h3 className="text-2xl font-display mb-6">MargNet Architecture</h3>
              <p className="text-ink-600 mb-8 max-w-2xl">
                A custom 4-stage convolutional neural network built from scratch. Designed to be lightweight and fast, using batch normalization and global average pooling.
              </p>
              
              <div className="flex flex-wrap items-center gap-4 mb-12">
                <ArchitectureBlock name="Input" details="64x64 RGB" shape="3 × 64 × 64" />
                <span className="text-lavender-300">→</span>
                <ArchitectureBlock name="Stage 1" details="32 filters" shape="32 × 32 × 32" />
                <span className="text-lavender-300">→</span>
                <ArchitectureBlock name="Stage 2" details="64 filters" shape="64 × 16 × 16" />
                <span className="text-lavender-300">→</span>
                <ArchitectureBlock name="Stage 3" details="128 filters" shape="128 × 8 × 8" />
                <span className="text-lavender-300">→</span>
                <ArchitectureBlock name="Stage 4" details="256 filters" shape="256 × 4 × 4" />
                <span className="text-lavender-300">→</span>
                <ArchitectureBlock name="GAP" details="Global Pool" shape="256" />
                <span className="text-lavender-300">→</span>
                <ArchitectureBlock name="Head" details="Dense K" shape="K classes" />
              </div>

              <h4 className="font-semibold mb-4 text-ink-900">Building Blocks</h4>
              <div className="grid md:grid-cols-3 gap-4">
                <Card className="p-4 bg-white"><h5 className="font-semibold text-sm mb-2 text-purple-500">Convolution</h5><p className="text-xs text-ink-600">Extracts spatial features using 3x3 sliding filters.</p></Card>
                <Card className="p-4 bg-white"><h5 className="font-semibold text-sm mb-2 text-purple-500">ReLU</h5><p className="text-xs text-ink-600">Introduces non-linearity to learn complex patterns.</p></Card>
                <Card className="p-4 bg-white"><h5 className="font-semibold text-sm mb-2 text-purple-500">Batch Norm</h5><p className="text-xs text-ink-600">Stabilizes and speeds up training by normalizing activations.</p></Card>
                <Card className="p-4 bg-white"><h5 className="font-semibold text-sm mb-2 text-purple-500">Max Pooling</h5><p className="text-xs text-ink-600">Reduces spatial dimensions, providing translation invariance.</p></Card>
                <Card className="p-4 bg-white"><h5 className="font-semibold text-sm mb-2 text-purple-500">Global Average Pooling</h5><p className="text-xs text-ink-600">Reduces feature maps to a single vector, preventing overfitting.</p></Card>
                <Card className="p-4 bg-white"><h5 className="font-semibold text-sm mb-2 text-purple-500">Dropout</h5><p className="text-xs text-ink-600">Randomly zeroes elements to improve generalization.</p></Card>
              </div>
            </Card>
          </Reveal>
        )}

        {tab === 'resnet' && (
          <Reveal>
            <Card className="p-8 bg-cream-50">
              <h3 className="text-2xl font-display mb-6">ResNet50 Architecture</h3>
              <p className="text-ink-600 mb-8 max-w-2xl">
                A 50-layer deep network utilizing residual connections to solve the vanishing gradient problem. Pretrained on ImageNet.
              </p>
              
              <div className="bg-white p-6 rounded-xl border border-lavender-200 mb-8">
                <h4 className="font-semibold mb-3 text-ochre">Two-Stage Fine-Tuning</h4>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <strong className="block text-sm mb-1 text-ink-900">Stage 1: Frozen Backbone</strong>
                    <p className="text-xs text-ink-600">Only the new classification head is trained. The feature extractor remains unchanged, acting as a robust feature extractor.</p>
                  </div>
                  <div>
                    <strong className="block text-sm mb-1 text-ink-900">Stage 2: Full Fine-Tuning</strong>
                    <p className="text-xs text-ink-600">All layers are unfrozen and trained with a very low learning rate to adapt the entire network to Indian traffic signs.</p>
                  </div>
                </div>
              </div>
            </Card>
          </Reveal>
        )}

        {tab === 'sidebyside' && (
          <Reveal>
            <Card className="p-0 overflow-hidden">
              <table className="w-full text-left">
                <thead className="bg-lavender-100/50">
                  <tr>
                    <th className="p-4 font-medium text-ink-600 text-sm">Feature</th>
                    <th className="p-4 font-semibold text-purple-700">MargNet</th>
                    <th className="p-4 font-semibold text-ochre">ResNet50</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-lavender-100">
                  <tr><td className="p-4 text-sm">Input Size</td><td className="p-4 font-mono text-sm">64x64</td><td className="p-4 font-mono text-sm">224x224</td></tr>
                  <tr><td className="p-4 text-sm">Pretrained</td><td className="p-4 text-sm">No (Scratch)</td><td className="p-4 text-sm">Yes (ImageNet)</td></tr>
                  <tr><td className="p-4 text-sm">Parameters</td><td className="p-4 font-mono text-sm">~1.2M</td><td className="p-4 font-mono text-sm">23.6M</td></tr>
                  <tr><td className="p-4 text-sm">Depth</td><td className="p-4 text-sm">10 layers</td><td className="p-4 text-sm">50 layers</td></tr>
                </tbody>
              </table>
            </Card>
          </Reveal>
        )}
      </Section>
    </div>
  );
}
