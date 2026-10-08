import React from 'react';
import { motion } from 'motion/react';
import { Link } from 'react-router-dom';
import { Database, FileCode, CheckCircle, Code } from 'lucide-react';

export default function About() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="max-w-4xl mx-auto space-y-12 pb-16"
    >
      <div className="space-y-4">
        <h1 className="text-4xl md:text-5xl font-serif text-ink-900 tracking-tight">
          About MargDrishti
        </h1>
        <p className="text-xl text-ink-600 max-w-[70ch]">
          MargDrishti (मार्गदृष्टि), meaning "vision of the road", is a college Data Science lab project comparing a custom CNN built from scratch with a fine-tuned ResNet50 for Indian traffic sign classification.
        </p>
      </div>

      <section className="space-y-6">
        <h2 className="text-2xl font-serif text-ink-900">Project Methodology</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {[
            { title: "1. Data Acquisition", desc: "4,354 images from Roboflow, 75 classes of Indian traffic signs." },
            { title: "2. Preprocessing", desc: "Cropping bounding boxes, resizing to 64x64 (Custom CNN) and 224x224 (ResNet50)." },
            { title: "3. Custom CNN Iteration", desc: "Five ablation steps starting from a simple CNN up to MargNet with augmentations and class weights." },
            { title: "4. ResNet50 Transfer Learning", desc: "Frozen backbone training, followed by full fine-tuning." },
            { title: "5. Rigorous Evaluation", desc: "Fair comparison on an unseen test set using multiple metrics." },
            { title: "6. Web Application", desc: "Local React/FastAPI dashboard to present results and run live inference." }
          ].map((step, i) => (
            <div key={i} className="p-4 bg-white rounded-2xl border border-lavender-200 shadow-sm">
              <h3 className="font-medium text-ink-900 mb-1">{step.title}</h3>
              <p className="text-sm text-ink-600">{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-serif text-ink-900">Fairness Rules</h2>
        <ul className="space-y-3">
          {[
            "Same crops, splits (train/valid/test), and random seed (42) for all models.",
            "Test set used exactly once per model, never for hyperparameter tuning.",
            "Identical metrics computation for all models."
          ].map((rule, i) => (
            <li key={i} className="flex gap-3 text-ink-600">
              <CheckCircle className="w-5 h-5 text-sage shrink-0" />
              <span>{rule}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-serif text-ink-900">Tech Stack</h2>
        <div className="flex flex-wrap gap-2">
          {["Python 3", "PyTorch", "FastAPI", "React", "Vite", "Tailwind CSS v4", "Recharts", "Motion"].map((tech) => (
            <div key={tech} className="px-3 py-1 bg-lavender-100 text-purple-700 rounded-full text-sm font-medium">
              {tech}
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-serif text-ink-900">Dataset Credit</h2>
        <p className="text-ink-600">
          Indian Traffic SignBoards from Roboflow Universe (by MAJOR PROJECT). MIT licence.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-serif text-ink-900">How to Reproduce</h2>
        <div className="bg-ink-900 rounded-2xl p-6 text-cream-50 font-mono text-sm overflow-x-auto">
          <pre className="space-y-2">
            <code>python -m venv .venv</code>{"\n"}
            <code>.\.venv\Scripts\activate</code>{"\n"}
            <code>pip install -r requirements.txt</code>{"\n"}
            <code>python -m scripts.run_all_reports</code>{"\n"}
          </pre>
        </div>
      </section>
    </motion.div>
  );
}
