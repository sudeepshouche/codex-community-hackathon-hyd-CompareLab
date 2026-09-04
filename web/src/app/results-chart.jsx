"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function ResultsChart({ data, showB = false }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid vertical={false} stroke="#e4e4e7" />
        <XAxis dataKey="time" tickFormatter={(v) => `${v}s`} tickLine={false} axisLine={false} />
        <YAxis domain={[0, 100]} tickLine={false} axisLine={false} width={30} />
        <RechartsTooltip />
        <Line type="monotone" dataKey="a" stroke="#18181b" strokeWidth={2} dot={false} />
        {showB && <Line type="monotone" dataKey="b" stroke="hsl(var(--tenant-primary))" strokeWidth={2} dot={false} />}
      </LineChart>
    </ResponsiveContainer>
  );
}
