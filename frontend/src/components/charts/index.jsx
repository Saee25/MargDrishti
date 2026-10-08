import React, { useContext } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  LineChart, Line, ScatterChart, Scatter, Cell
} from 'recharts';
import { motion } from 'motion/react';
import { MotionContext } from '../../lib/motion';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-cream-50 border border-lavender-200 p-3 rounded-lg shadow-soft">
        {label && <p className="text-sm font-semibold text-ink-900 mb-1">{label}</p>}
        {payload.map((entry, index) => (
          <p key={index} className="text-xs text-ink-600 font-mono">
            <span style={{ color: entry.color }}>■ </span>
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const CustomLegend = ({ payload }) => {
  return (
    <div className="flex flex-wrap justify-center gap-4 mt-4">
      {payload.map((entry, index) => (
        <div key={index} className="flex items-center gap-1.5 text-xs text-ink-600 bg-lavender-100 px-2 py-1 rounded-full border border-lavender-200">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
          <span>{entry.value}</span>
        </div>
      ))}
    </div>
  );
};

function ChartWrapper({ title, caption, summary, children }) {
  const { reduced } = useContext(MotionContext);
  return (
    <div className="w-full">
      <div className="mb-4">
        <h4 className="font-semibold text-ink-900">{title}</h4>
        {caption && <p className="text-xs text-ink-600">{caption}</p>}
        <span className="sr-only">{summary}</span>
      </div>
      <motion.div
        className="w-full h-[300px]"
        initial={reduced ? false : { opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      >
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
}

const defaultAxisProps = {
  tick: { fontFamily: 'DM Sans', fontSize: 12, fill: '#5B5470' },
  axisLine: false,
  tickLine: false,
};

const defaultGridProps = {
  strokeDasharray: '0',
  vertical: false,
  stroke: '#F4ECC6',
  strokeWidth: 1,
};

export function BarChartV({ data, keys, colors, title, caption, summary }) {
  return (
    <ChartWrapper title={title} caption={caption} summary={summary}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid {...defaultGridProps} />
        <XAxis dataKey="name" {...defaultAxisProps} />
        <YAxis {...defaultAxisProps} />
        <RechartsTooltip content={<CustomTooltip />} />
        <Legend content={<CustomLegend />} />
        {keys.map((key, i) => (
          <Bar key={key} dataKey={key} fill={colors[i]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ChartWrapper>
  );
}

export function BarChartH({ data, keys, colors, title, caption, summary }) {
  return (
    <ChartWrapper title={title} caption={caption} summary={summary}>
      <BarChart data={data} layout="vertical" margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid {...defaultGridProps} horizontal={false} vertical={true} />
        <XAxis type="number" {...defaultAxisProps} />
        <YAxis dataKey="name" type="category" width={100} {...defaultAxisProps} />
        <RechartsTooltip content={<CustomTooltip />} />
        <Legend content={<CustomLegend />} />
        {keys.map((key, i) => (
          <Bar key={key} dataKey={key} fill={colors[i]} radius={[0, 4, 4, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ChartWrapper>
  );
}

export function LineChartMulti({ data, keys, colors, title, caption, summary }) {
  return (
    <ChartWrapper title={title} caption={caption} summary={summary}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid {...defaultGridProps} />
        <XAxis dataKey="name" {...defaultAxisProps} />
        <YAxis {...defaultAxisProps} />
        <RechartsTooltip content={<CustomTooltip />} />
        <Legend content={<CustomLegend />} />
        {keys.map((key, i) => (
          <Line key={key} type="monotone" dataKey={key} stroke={colors[i]} strokeWidth={2} dot={{ r: 3, fill: colors[i] }} isAnimationActive={false} />
        ))}
      </LineChart>
    </ChartWrapper>
  );
}

export function ScatterPlot({ data, xAxis, yAxis, color, title, caption, summary }) {
  return (
    <ChartWrapper title={title} caption={caption} summary={summary}>
      <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid {...defaultGridProps} />
        <XAxis type="number" dataKey={xAxis} name={xAxis} {...defaultAxisProps} />
        <YAxis type="number" dataKey={yAxis} name={yAxis} {...defaultAxisProps} />
        <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
        <Scatter name="Data" data={data} fill={color} isAnimationActive={false}>
          {data.map((entry, index) => <Cell key={`cell-${index}`} fill={color} />)}
        </Scatter>
      </ScatterChart>
    </ChartWrapper>
  );
}

export function Histogram({ data, dataKey, color, title, caption, summary, bins = 10 }) {
  return (
    <ChartWrapper title={title} caption={caption} summary={summary}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid {...defaultGridProps} />
        <XAxis dataKey="name" {...defaultAxisProps} />
        <YAxis {...defaultAxisProps} />
        <RechartsTooltip content={<CustomTooltip />} />
        <Bar dataKey={dataKey} fill={color} radius={[4, 4, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ChartWrapper>
  );
}
