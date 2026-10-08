import React from 'react';
import { motion } from 'motion/react';
import { ArrowRight, ChevronRight, BarChart2 } from 'lucide-react';
import { Reveal, CountUp, staggerContainer, fadeRise } from '../lib/motion';
import { Section, Card, StatCard, Button, EmptyState, Skeleton } from '../components/ui';
import { useApi } from '../hooks/useApi';
import { getMetricsSummary, getDatasetStats, getSamples } from '../lib/api';
import { Link } from 'react-router-dom';

function HeroSamples({ samples, loading }) {
  if (loading) {
    return (
      <div className="grid grid-cols-3 gap-2 p-4 md:p-8 relative h-64 md:h-full">
        <Skeleton className="rounded-xl w-full h-full" />
        <Skeleton className="rounded-xl w-full h-full" />
        <Skeleton className="rounded-xl w-full h-full" />
      </div>
    );
  }

  if (!samples || samples.length === 0) return null;

  return (
    <div className="relative w-full h-64 md:h-96">
      <div className="absolute inset-0 bg-cream-100 rounded-3xl -rotate-2 scale-95 shadow-soft z-0 flex items-center justify-center overflow-hidden">
        <div className="w-64 h-64 rounded-full bg-lavender-100/50 blur-3xl" />
      </div>
      
      <motion.div 
        variants={staggerContainer}
        initial="hidden"
        animate="show"
        className="relative z-10 w-full h-full p-4 md:p-8 grid grid-cols-3 grid-rows-3 gap-2 md:gap-3 lg:gap-4"
      >
        {samples.slice(0, 9).map((sample, i) => (
          <motion.div
            key={i}
            variants={fadeRise}
            whileHover={{ y: -4, scale: 1.02 }}
            className={`rounded-xl md:rounded-2xl overflow-hidden shadow-soft border border-lavender-200 bg-white
              ${i === 4 ? 'col-span-2 row-span-2' : ''}
              ${i === 0 ? 'rounded-tl-3xl' : ''}
              ${i === 2 ? 'rounded-tr-3xl' : ''}
              ${i === 6 ? 'rounded-bl-3xl' : ''}
              ${i === 8 ? 'rounded-br-3xl' : ''}
            `}
          >
            <img src={sample.url} alt={`Traffic sign ${sample.class_id}`} className="w-full h-full object-cover" />
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}

function PipelineFlow() {
  const steps = [
    { label: 'Road images', desc: '4,354 raw Indian scenes' },
    { label: 'Crop signs', desc: 'Bounding box extraction' },
    { label: 'Preprocess', desc: 'Augment & balance' },
    { label: 'Train two models', desc: 'Custom CNN vs ResNet50' },
    { label: 'Compare', desc: 'Accuracy & efficiency' }
  ];

  return (
    <Reveal className="w-full py-12 overflow-x-auto pb-4">
      <div className="flex items-start min-w-[800px]">
        {steps.map((step, i) => (
          <React.Fragment key={i}>
            <div className="flex flex-col items-center flex-1 text-center relative z-10">
              <div className="w-12 h-12 rounded-full bg-cream-50 border-2 border-purple-500 text-purple-700 flex items-center justify-center font-display font-semibold text-xl mb-4 shadow-soft">
                {i + 1}
              </div>
              <h4 className="font-semibold text-ink-900 mb-1">{step.label}</h4>
              <p className="text-xs text-ink-600 px-2">{step.desc}</p>
            </div>
            {i < steps.length - 1 && (
              <div className="flex-1 mt-6 relative h-[2px]">
                <div className="absolute inset-0 bg-lavender-200" />
                <motion.div 
                  className="absolute inset-0 bg-purple-500 origin-left"
                  initial={{ scaleX: 0 }}
                  whileInView={{ scaleX: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8, delay: i * 0.2, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </Reveal>
  );
}

export default function Home() {
  const { data: stats, loading: statsLoading } = useApi(getDatasetStats);
  const { data: metrics, loading: metricsLoading } = useApi(getMetricsSummary);
  const { data: samples, loading: samplesLoading } = useApi(getSamples, 9);

  const margnetAcc = metrics?.models?.['margnet-v5']?.metrics?.accuracy || 0;
  const resnetAcc = metrics?.models?.['resnet50-finetuned']?.metrics?.accuracy || 0;

  const showStats = stats || metrics;

  return (
    <div className="flex flex-col gap-16 md:gap-24">
      {/* Hero */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center min-h-[60vh]">
        <Reveal>
          <div className="text-purple-500 font-semibold tracking-wider uppercase text-sm mb-4">Indian Traffic Sign Recognition</div>
          <h1 className="text-5xl md:text-6xl lg:text-7xl mb-6 leading-[1.1]">Reading the road,<br/>one sign at a time.</h1>
          <p className="text-xl md:text-2xl text-ink-600 leading-relaxed mb-10 max-w-lg">
            How close can a small custom CNN get to a fine-tuned ResNet50 on Indian traffic signs? 
            And at what fraction of the cost?
          </p>
          <div className="flex flex-wrap gap-4">
            <Button as="Link" to="/demo" variant="primary">
              Try the live demo <ChevronRight size={18} />
            </Button>
            <Button as="Link" to="/comparison" variant="secondary">
              See the comparison
            </Button>
          </div>
        </Reveal>
        <Reveal delay={0.2} className="relative w-full">
          <HeroSamples samples={samples} loading={samplesLoading} />
        </Reveal>
      </section>

      {/* Stats */}
      {showStats ? (
        <Reveal>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
            <StatCard 
              label="Classes" 
              value={statsLoading ? <Skeleton className="w-16 h-10" /> : <CountUp to={stats?.stats?.num_classes || 0} />} 
              caption="Unique Indian traffic signs" 
            />
            <StatCard 
              label="Cropped Signs" 
              value={statsLoading ? <Skeleton className="w-24 h-10" /> : <CountUp to={stats?.stats?.total_crops || 0} />} 
              caption="From 4,354 source images" 
            />
            <StatCard 
              label="MargNet Accuracy" 
              value={metricsLoading ? <Skeleton className="w-24 h-10" /> : <><CountUp to={Math.round(margnetAcc * 100)} />%</>} 
              caption="Test set macro F1" 
              className="border-purple-200 bg-purple-50/50"
            />
            <StatCard 
              label="ResNet50 Accuracy" 
              value={metricsLoading ? <Skeleton className="w-24 h-10" /> : <><CountUp to={Math.round(resnetAcc * 100)} />%</>} 
              caption="Test set macro F1" 
              className="border-ochre/20 bg-ochre/5"
            />
          </div>
        </Reveal>
      ) : (
        <EmptyState 
          icon={BarChart2}
          title="No data available yet" 
          description="Run the data processing and training scripts to generate the models and stats."
          command="python -m scripts.run_all_reports"
        />
      )}

      {/* The Question */}
      <Reveal className="max-w-4xl mx-auto text-center py-8">
        <blockquote className="font-display text-3xl md:text-4xl text-ink-900 leading-snug mb-6">
          "Can a model with less than 2% of ResNet50's parameters accurately classify Indian traffic signs on a CPU in real-time?"
        </blockquote>
        <p className="text-ink-600 text-lg">
          This project explores the trade-offs between model size, inference speed, and accuracy 
          in the context of autonomous driving assistance systems in developing regions.
        </p>
      </Reveal>

      {/* Pipeline */}
      <Section title="The Pipeline" className="bg-cream-100/30 rounded-3xl p-8 border border-lavender-200">
        <PipelineFlow />
      </Section>

      {/* Models */}
      <Section title="The Contenders" lead="Two very different architectures trained on the exact same dataset splits.">
        {metricsLoading ? (
          <div className="grid md:grid-cols-2 gap-8">
            <Skeleton className="h-64 rounded-2xl" />
            <Skeleton className="h-64 rounded-2xl" />
          </div>
        ) : metrics?.models ? (
          <div className="grid md:grid-cols-2 gap-8">
            {/* MargNet */}
            <Card className="p-8 border-t-4 border-t-purple-500">
              <h3 className="text-2xl font-display font-semibold mb-2">MargNet</h3>
              <p className="text-ink-600 mb-6 h-12">A lightweight custom CNN designed specifically for road sign classification.</p>
              
              <div className="space-y-4 mb-8">
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Parameters</span>
                  <span className="font-mono">{metrics.models['margnet-v5']?.metrics?.params?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Size</span>
                  <span className="font-mono">{metrics.models['margnet-v5']?.metrics?.model_size_mb?.toFixed(1) || 'N/A'} MB</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Inference (CPU)</span>
                  <span className="font-mono">{metrics.models['margnet-v5']?.metrics?.latency_cpu_ms?.toFixed(1) || 'N/A'} ms</span>
                </div>
              </div>
              
              <Button as="Link" to="/models" variant="quiet" className="w-full -ml-4">
                View model details <ArrowRight size={16} />
              </Button>
            </Card>

            {/* ResNet50 */}
            <Card className="p-8 border-t-4 border-t-ochre">
              <h3 className="text-2xl font-display font-semibold mb-2">ResNet50</h3>
              <p className="text-ink-600 mb-6 h-12">The industry standard, fine-tuned from ImageNet weights.</p>
              
              <div className="space-y-4 mb-8">
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Parameters</span>
                  <span className="font-mono">{metrics.models['resnet50-finetuned']?.metrics?.params?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Size</span>
                  <span className="font-mono">{metrics.models['resnet50-finetuned']?.metrics?.model_size_mb?.toFixed(1) || 'N/A'} MB</span>
                </div>
                <div className="flex justify-between items-center border-b border-lavender-100 pb-2">
                  <span className="text-ink-600 text-sm">Inference (CPU)</span>
                  <span className="font-mono">{metrics.models['resnet50-finetuned']?.metrics?.latency_cpu_ms?.toFixed(1) || 'N/A'} ms</span>
                </div>
              </div>
              
              <Button as="Link" to="/models" variant="quiet" className="w-full -ml-4 !text-ochre hover:!bg-ochre/10">
                View model details <ArrowRight size={16} />
              </Button>
            </Card>
          </div>
        ) : (
           <EmptyState 
            icon={BarChart2}
            title="Models not evaluated" 
            description="Run the evaluation scripts to see model details."
            command="python -m scripts.run_all_reports"
          />
        )}
      </Section>

      {/* CTA */}
      <Reveal className="bg-lavender-100 rounded-3xl p-10 md:p-16 text-center border border-lavender-200 shadow-inner">
        <h2 className="text-3xl md:text-4xl mb-4">Try it yourself</h2>
        <p className="text-lg text-ink-600 mb-8 max-w-2xl mx-auto">
          Upload an image of an Indian traffic sign or use one from our test set to see how both models perform in real-time.
        </p>
        <Button as="Link" to="/demo" variant="primary" className="text-base px-8 py-4">
          Open Live Demo
        </Button>
      </Reveal>
    </div>
  );
}
