import React, { useState, useRef } from 'react';
import { Upload, AlertCircle, RefreshCw } from 'lucide-react';
import { Section, Card, Container, Button, Skeleton } from '../components/ui';
import { predict } from '../lib/api';

export default function LiveDemo() {
  const [modelType, setModelType] = useState('both');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = async (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResults(null);
    setError(null);
    setLoading(true);

    try {
      if (modelType === 'both') {
        const [margnet, resnet] = await Promise.all([
          predict('margnet-v5', selectedFile).catch(e => ({ error: e.message })),
          predict('resnet50-finetuned', selectedFile).catch(e => ({ error: e.message }))
        ]);
        setResults({ margnet, resnet });
      } else {
        const res = await predict(modelType, selectedFile);
        setResults({ [modelType]: res });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <Section title="Live Demo" lead="Test the models on your own cropped traffic sign images.">
        <div className="grid lg:grid-cols-12 gap-8">
          
          <div className="lg:col-span-4 flex flex-col gap-6">
            <Card className="p-6">
              <div className="flex bg-lavender-100 p-1 rounded-lg mb-6 w-full">
                {[{id: 'both', label: 'Both'}, {id: 'margnet-v5', label: 'MargNet'}, {id: 'resnet50-finetuned', label: 'ResNet50'}].map(m => (
                  <button
                    key={m.id}
                    onClick={() => setModelType(m.id)}
                    className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${modelType === m.id ? 'bg-white shadow-sm text-purple-700' : 'text-ink-600'}`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              <div 
                className="border-2 border-dashed border-lavender-200 rounded-xl p-8 text-center bg-cream-50 hover:bg-lavender-100/50 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => e.preventDefault()}
                onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
              >
                <Upload size={32} className="mx-auto text-lavender-300 mb-3" />
                <p className="text-sm font-medium text-ink-900 mb-1">Upload a cropped sign</p>
                <p className="text-xs text-ink-400">Drag & drop or click to browse</p>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept="image/jpeg,image/png"
                  onChange={e => handleFile(e.target.files[0])}
                />
              </div>
              <p className="text-xs text-ink-400 mt-4 text-center">Note: The models expect a tightly cropped traffic sign, not a full road image.</p>
            </Card>
          </div>

          <div className="lg:col-span-8">
            <Card className="p-6 min-h-[400px] flex flex-col bg-cream-50">
              {!preview && !loading && !results && !error && (
                <div className="flex-1 flex flex-col items-center justify-center text-ink-400">
                  <div className="w-16 h-16 rounded-full bg-lavender-100 flex items-center justify-center mb-4">
                    <AlertCircle size={24} className="text-lavender-300" />
                  </div>
                  <p>Upload an image to see predictions</p>
                </div>
              )}

              {preview && (
                <div className="mb-8 self-center">
                  <img src={preview} alt="Upload preview" className="w-32 h-32 object-cover rounded-xl shadow-soft border border-lavender-200" />
                </div>
              )}

              {loading && (
                <div className="grid md:grid-cols-2 gap-6 w-full">
                  <Skeleton className="h-48 rounded-xl" />
                  {modelType === 'both' && <Skeleton className="h-48 rounded-xl" />}
                </div>
              )}

              {error && (
                <div className="bg-rose/10 text-rose p-4 rounded-xl text-sm flex items-start gap-3">
                  <AlertCircle size={18} className="mt-0.5" />
                  <div className="flex-1">
                    <p className="font-semibold mb-1">Prediction Failed</p>
                    <p>{error}</p>
                  </div>
                </div>
              )}

              {results && !loading && (
                <div className="grid md:grid-cols-2 gap-6 w-full" aria-live="polite">
                  {results.margnet && (
                    <div className="bg-white p-5 rounded-xl border border-lavender-200 shadow-sm border-t-4 border-t-purple-500">
                      <h4 className="font-semibold text-purple-700 mb-4 text-sm uppercase tracking-wider">MargNet</h4>
                      {results.margnet.error ? (
                        <p className="text-xs text-rose">{results.margnet.error}</p>
                      ) : (
                        <div>
                          <p className="text-2xl font-display mb-1">{results.margnet.prediction}</p>
                          <p className="text-sm text-ink-600 mb-4 font-mono">{(results.margnet.confidence * 100).toFixed(1)}% confidence</p>
                          <p className="text-xs text-ink-400 font-mono mt-4 pt-4 border-t border-lavender-100">Latency: {results.margnet.latency_ms?.toFixed(1)}ms</p>
                        </div>
                      )}
                    </div>
                  )}

                  {results.resnet && (
                    <div className="bg-white p-5 rounded-xl border border-lavender-200 shadow-sm border-t-4 border-t-ochre">
                      <h4 className="font-semibold text-ochre mb-4 text-sm uppercase tracking-wider">ResNet50</h4>
                      {results.resnet.error ? (
                        <p className="text-xs text-rose">{results.resnet.error}</p>
                      ) : (
                        <div>
                          <p className="text-2xl font-display mb-1">{results.resnet.prediction}</p>
                          <p className="text-sm text-ink-600 mb-4 font-mono">{(results.resnet.confidence * 100).toFixed(1)}% confidence</p>
                          <p className="text-xs text-ink-400 font-mono mt-4 pt-4 border-t border-lavender-100">Latency: {results.resnet.latency_ms?.toFixed(1)}ms</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </Card>
          </div>
          
        </div>
      </Section>
    </div>
  );
}
