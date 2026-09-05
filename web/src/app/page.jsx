"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckCircle2Icon,
  CheckIcon,
  CircleDashedIcon,
  CommandIcon,
  EyeIcon,
  FileAudioIcon,
  FileTextIcon,
  FileVideoIcon,
  InfoIcon,
  Layers2Icon,
  LoaderCircleIcon,
  ScanSearchIcon,
  PanelRightCloseIcon,
  PanelRightOpenIcon,
  PlayIcon,
  SparklesIcon,
  UploadIcon,
  ZapIcon,
} from "lucide-react";

const ResultsChart = dynamic(() => import("./results-chart"), {
  ssr: false,
  loading: () => <div className="h-full w-full animate-pulse rounded-xl bg-zinc-100" />,
});

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const API_ENDPOINT = "/api/analyze";
const TOAST_TIMEOUT_MS = 5000;

const PROGRESS_PHASES = [
  { key: "queued", label: "Starting", percent: 8 },
  { key: "preparing", label: "Preparing files", percent: 24 },
  { key: "scoring", label: "Reviewing each version", percent: 62 },
  { key: "comparing", label: "Building your results", percent: 90 },
  { key: "complete", label: "Ready", percent: 100 },
];

const STIMULUS_MODES = [
  { value: "video", label: "Video", icon: FileVideoIcon, accept: "video/*,image/*" },
  { value: "audio", label: "Audio", icon: FileAudioIcon, accept: "audio/*" },
  { value: "text", label: "Text", icon: FileTextIcon, accept: ".txt,text/plain" },
];

const ROW_METADATA = {
  overall: { desc: "Best overall read across all tracked signals.", insight: "has the stronger overall read, meaning it is more likely to hold attention across the full piece than" },
  attention: { desc: "How well this version grabs and holds focus.", insight: "does a better job of pulling focus and keeping the viewer mentally locked in than" },
  memory: { desc: "How likely this version is to leave a lasting impression.", insight: "leaves a more memorable pattern, meaning more of the message survives than" },
  emotion: { desc: "How strongly this version seems to land emotionally.", insight: "creates a stronger emotional signal, feeling more moving than merely seen compared to" },
  reward: { desc: "How satisfying or engaging this version feels.", insight: "feels more satisfying moment to moment, making it less easy to skip than" },
  novelty: { desc: "How fresh or distinctive this version feels.", insight: "feels a bit fresher, helping it stand out before the audience tunes it out, unlike" },
  opening: { desc: "How strongly this version starts.", insight: "starts stronger, giving people a better reason to stay in the first few seconds than" },
  middle: { desc: "How well this version holds up in the middle.", insight: "holds together better in the middle, where weaker creatives often sag, compared to" },
  closing: { desc: "How strongly this version finishes.", insight: "finishes stronger, leaving a clearer final impression than" },
  peak: { desc: "The biggest standout moment in the piece.", insight: "reaches a higher high point, delivering a more noticeable standout moment than" },
  spread: { desc: "How much the response rises and falls over time.", insight: "shows more dynamic movement across the piece compared to the flatter execution of" },
  consistency: { desc: "How steady the response feels from moment to moment.", insight: "is steadier from moment to moment, providing a less uneven experience than" },
};

// ── Utilities ──────────────────────────────────────────────

const fmt = (v, d = 3) => (v == null || isNaN(v) ? "—" : Number(v).toFixed(d));
const fmtSec = (v) => (v == null || isNaN(v) ? "—" : `${Number(v).toFixed(1)}s`);
const fmtMs = (v) => (v == null || isNaN(v) ? "—" : `${Math.round(Number(v))} ms`);
const versionLabel = (label, fb = "Version") => String(label ?? fb).trim().replace(/^stimulus/i, "Version") || fb;
const toNumberOrNull = (v) => (Number.isFinite(Number(v)) ? Number(v) : null);

function fmtBytes(v) {
  if (!Number.isFinite(v)) return "—";
  if (v < 1024) return `${v} B`;
  return v < 1024 * 1024 ? `${(v / 1024).toFixed(1)} KB` : `${(v / (1024 * 1024)).toFixed(1)} MB`;
}

function normalizeCurvesForChart(curves) {
  const map = new Map();
  curves.forEach((pts, i) => {
    const max = Math.max(...pts.map((p) => Number(p.score ?? 0)), 1);
    pts.forEach((p) => {
      const t = Number(p.midpoint_s ?? p.time_s ?? 0);
      const e = map.get(t) ?? { time: t, a: null, b: null };
      e[i === 0 ? "a" : "b"] = (Number(p.score ?? 0) / max) * 100;
      map.set(t, e);
    });
  });
  return [...map.values()].sort((a, b) => a.time - b.time);
}

function buildRowInsight(key, label, winner, aValue, bValue, labelA, labelB) {
  if (aValue == null || bValue == null) return "This row adds technical context, but the result is not easy to translate cleanly.";
  if (winner === "tie") return `${labelA} and ${labelB} are effectively even on this measure.`;

  const gapPct = (Math.abs(aValue - bValue) / Math.max((Math.abs(aValue) + Math.abs(bValue)) / 2, 0.0001)) * 100;
  const closeness = gapPct < 5 ? "It is a close call, but" : "There is a clearer gap here;";
  const betterLabel = winner === "A" ? labelA : labelB;
  const otherLabel = winner === "A" ? labelB : labelA;
  
  const metaKey = Object.keys(ROW_METADATA).find((k) => label.toLowerCase().includes(k) || key.toLowerCase().includes(k));
  const insightText = ROW_METADATA[metaKey]?.insight ?? "comes out ahead on this measure over";

  return `${closeness} ${betterLabel} ${insightText} ${otherLabel}.`;
}

function buildComparisonRows({ comparison, stimulusA, stimulusB }) {
  if (!comparison || !stimulusA || !stimulusB) return [];
  const lA = versionLabel(stimulusA.label, "Version A");
  const lB = versionLabel(stimulusB.label, "Version B");

  const buildRow = (key, label, aScore, bScore, aNote, bNote, winner) => {
    const metaKey = Object.keys(ROW_METADATA).find((k) => label.toLowerCase().includes(k) || key.toLowerCase().includes(k));
    return {
      key, label, a: fmt(aScore), b: fmt(bScore), aNote, bNote, winner: winner === "tie" ? "tie" : winner,
      description: ROW_METADATA[metaKey]?.desc ?? "A comparison point for the two versions.",
      insight: buildRowInsight(key, label, winner, toNumberOrNull(aScore), toNumberOrNull(bScore), lA, lB),
    };
  };

  const rows = [];
  if (comparison.engagement?.overall) {
    const { a, b, winner } = comparison.engagement.overall;
    rows.push(buildRow("overall-score", "Overall score", a, b, stimulusA.scorecard?.overall_band ?? "—", stimulusB.scorecard?.overall_band ?? "—", winner));
  }
  
  (comparison.engagement?.systems ?? []).forEach((sys) => {
    const aNote = stimulusA.scorecard?.systems?.find((s) => s.key === sys.key)?.readout ?? "—";
    const bNote = stimulusB.scorecard?.systems?.find((s) => s.key === sys.key)?.readout ?? "—";
    rows.push(buildRow(`system-${sys.key}`, sys.label, sys.a, sys.b, aNote, bNote, sys.winner));
  });

  (comparison.metrics ?? []).forEach((m) => {
    rows.push(buildRow(`metric-${m.name}`, m.label ?? m.name, m.a, m.b, "score", "score", m.winner));
  });

  return rows;
}

// ── Hook: Object URL lifecycle ─────────────────────────────

function useObjectUrl(file) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!file) { setUrl(""); return; }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

// ── Shared UI Atoms ────────────────────────────────────────

const Card = ({ className = "", children, ...props }) => (
  <section className={`rounded-3xl border border-zinc-200/70 bg-white ${className}`} {...props}>{children}</section>
);

const SectionHeader = ({ tag, title, subtitle }) => (
  <div className="grid gap-1">
    {tag && <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[hsl(var(--tenant-primary))]">{tag}</p>}
    {title && <h2 className="text-base font-semibold">{title}</h2>}
    {subtitle && <p className="text-sm text-zinc-600">{subtitle}</p>}
  </div>
);

function StimulusCue({ label, previewUrl, mode, size = "md" }) {
  const showPreview = Boolean(previewUrl) && mode !== "audio" && mode !== "text";
  const initial = versionLabel(label, "Version").trim().charAt(0).toUpperCase() || "?";
  const classes = size === "sm" ? "size-8 rounded-lg" : "size-12 rounded-xl";

  return showPreview ? (
    <div className={`${classes} overflow-hidden border border-zinc-200/70 bg-[hsl(var(--tenant-primary))]/10`}>
      <img src={previewUrl} alt="preview" className="h-full w-full object-cover" />
    </div>
  ) : (
    <div className={`${classes} flex items-center justify-center border border-[hsl(var(--tenant-primary))]/20 bg-[hsl(var(--tenant-primary))]/15 text-[hsl(var(--tenant-primary))]`}>
      <span className="text-xs font-semibold">{initial}</span>
    </div>
  );
}

// ── Components ─────────────────────────────────────────────

function useIsDesktopRail() {
  // Tips render as a fixed side rail at xl+ and a drawer below that.
  const [isWide, setIsWide] = useState(null);
  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1280px)");
    const onChange = () => setIsWide(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);
  return isWide;
}

function TipsContent({ onOpenQuickStart }) {
  return (
    <div className="grid gap-4">
      <Card className="p-4 text-sm"><div className="font-semibold mb-1">Read order</div><p className="text-zinc-600">Summary, table, then deeper visuals.</p></Card>
      <Card className="p-4 text-sm"><div className="font-semibold mb-1">Best use</div><p className="text-zinc-600">Review two versions before you spend more budget.</p></Card>
      <Card className="p-4 text-sm"><div className="font-semibold mb-1">Fair compare</div><p className="text-zinc-600">Keep both versions in the same format and length range.</p></Card>
      <Card className="p-4"><Button variant="outline" className="w-full justify-start rounded-xl" onClick={onOpenQuickStart}><InfoIcon className="mr-2 size-4" /> Open quick start</Button></Card>
    </div>
  );
}

function QuickStartDialog({ open, onOpenChange }) {
  const steps = [
    {
      icon: Layers2Icon,
      eyebrow: "Step 1 · Setup",
      title: "Choose one format",
      headline: "Start with one shared content type.",
      body: "Video, audio, or text — pick the format both versions share so the comparison stays fair.",
      points: ["Video and image files", "Audio clips", "Pasted plain text"],
      tip: "Have one version as video and one as text? Export them to the same format first — mixed-format pairs can't be compared.",
    },
    {
      icon: EyeIcon,
      eyebrow: "Step 2 · Inputs",
      title: "Upload versions",
      headline: "Load the versions you want to test.",
      body: "Version A is required. Add Version B to unlock the side-by-side comparison with a winner per row.",
      points: ["One upload = single review", "Two uploads = head-to-head", "Previews appear instantly"],
      tip: "Upload the same file as both versions to sanity-check the pipeline before a real matchup.",
    },
    {
      icon: ScanSearchIcon,
      eyebrow: "Step 3 · Analysis",
      title: "Run the review",
      headline: "Let the app build the evidence.",
      body: "The TRIBE model reads each version and returns scores, a response curve over time, and plain-language notes.",
      points: ["Live progress while it works", "Response curve per version", "Repeat runs are instant (cached)"],
      tip: "The first analysis loads the model. After that, repeat runs of the same file finish in milliseconds.",
    },
    {
      icon: CommandIcon,
      eyebrow: "Step 4 · Decision",
      title: "Read the results",
      headline: "Treat this as decision support, not proof.",
      body: "Read the summary first, then the comparison table, then the curve. Close scores are effectively ties.",
      points: ["Overall score + band", "Winner per measure", "Key observations"],
      tip: "When two versions land within a few points, pick on taste or budget — not the tiny score gap.",
    },
  ];
  const [active, setActive] = useState(0);
  const step = steps[active];
  const isLast = active === steps.length - 1;

  useEffect(() => {
    if (open) setActive(0);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "ArrowRight") setActive((v) => Math.min(steps.length - 1, v + 1));
      if (e.key === "ArrowLeft") setActive((v) => Math.max(0, v - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden rounded-[28px] border-zinc-200/80 bg-white p-0 sm:max-w-3xl">
        <DialogHeader className="brand-gradient relative border-0 px-6 py-6 text-left">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(40rem_12rem_at_80%_-20%,rgba(255,255,255,0.35),transparent_60%)]" aria-hidden />
          <div className="relative flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-2xl bg-white/15 backdrop-blur"><SparklesIcon className="size-5 text-white" /></div>
            <div className="grid gap-0.5">
              <DialogTitle className="text-lg font-semibold text-white">Welcome to Compare Lab</DialogTitle>
              <DialogDescription className="text-sm text-white/80">Four steps from upload to a confident call.</DialogDescription>
            </div>
            <span className="ml-auto rounded-full bg-white/15 px-3 py-1 text-xs font-semibold text-white">{active + 1} / {steps.length}</span>
          </div>
        </DialogHeader>

        <div className="grid gap-5 px-6 py-5 md:grid-cols-[240px_minmax(0,1fr)]">
          <div className="relative grid content-start gap-2">
            <div className="absolute bottom-8 left-[27px] top-8 w-px bg-zinc-200" aria-hidden />
            {steps.map((s, i) => {
              const done = i < active;
              return (
                <button key={i} type="button" onClick={() => setActive(i)} className={`relative flex items-center gap-3 rounded-2xl border px-3.5 py-3 text-left transition-all ${active === i ? "border-transparent bg-[hsl(var(--tenant-primary))]/[0.07] shadow-sm ring-1 ring-[hsl(var(--tenant-primary))]/35" : "border-transparent hover:bg-zinc-50"}`}>
                  <span className={`z-10 flex size-7 shrink-0 items-center justify-center rounded-full text-xs font-bold ${active === i ? "brand-gradient text-white" : done ? "bg-[hsl(var(--tenant-primary))]/15 text-[hsl(var(--tenant-primary))]" : "border border-zinc-200 bg-white text-zinc-400"}`}>
                    {done ? <CheckIcon className="size-3.5" /> : i + 1}
                  </span>
                  <span className={`text-sm font-semibold ${active === i ? "text-zinc-950" : "text-zinc-500"}`}>{s.title}</span>
                </button>
              );
            })}
          </div>

          <div key={active} className="qs-enter grid content-start gap-4 rounded-2xl border border-zinc-200/70 bg-zinc-50/60 p-5">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[hsl(var(--tenant-primary))]">{step.eyebrow}</p>
              <h3 className="mt-1 text-xl font-semibold tracking-tight text-zinc-950">{step.headline}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-600">{step.body}</p>
            </div>
            <ul className="grid gap-1.5">
              {step.points.map((pt) => (
                <li key={pt} className="flex items-center gap-2 text-sm text-zinc-700">
                  <CheckCircle2Icon className="size-4 shrink-0 text-[hsl(var(--tenant-primary))]" /> {pt}
                </li>
              ))}
            </ul>
            <div className="rounded-2xl border border-[hsl(var(--tenant-primary))]/20 bg-white p-3.5">
              <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-[hsl(var(--tenant-primary))]"><ZapIcon className="size-3.5" /> Pro tip</div>
              <p className="mt-1 text-sm text-zinc-600">{step.tip}</p>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 border-t border-zinc-200/70 px-6 py-4">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="rounded-xl text-zinc-500" onClick={() => setActive(Math.max(0, active - 1))} disabled={active === 0}>
              <ArrowLeftIcon className="mr-1.5 size-4" /> Back
            </Button>
            {isLast ? (
              <Button size="sm" className="brand-gradient rounded-xl text-white shadow-md shadow-[hsl(243_75%_59%/0.25)] hover:opacity-90" onClick={() => onOpenChange(false)}>
                Start comparing <ArrowRightIcon className="ml-1.5 size-4" />
              </Button>
            ) : (
              <Button size="sm" className="brand-gradient rounded-xl text-white shadow-md shadow-[hsl(243_75%_59%/0.25)] hover:opacity-90" onClick={() => setActive(Math.min(steps.length - 1, active + 1))}>
                Next <ArrowRightIcon className="ml-1.5 size-4" />
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function StimulusPanel({ inputKey, title, description, mode, file, textValue, previewUrl, optional, onSelect, onTextChange }) {
  const currentMode = STIMULUS_MODES.find((m) => m.value === mode) || STIMULUS_MODES[0];
  return (
    <div className="grid gap-4 p-5">
      <div className="grid gap-1">
        <div className="flex items-center gap-2">
          <span className="flex size-7 items-center justify-center rounded-full border border-zinc-200 bg-zinc-50 text-xs font-semibold text-zinc-500">{title.slice(-1)}</span>
          <h2 className="text-base font-semibold">{title}</h2>
          {optional && <span className="text-xs text-zinc-500">Optional</span>}
        </div>
        <p className="text-sm text-zinc-600">{description}</p>
      </div>
      
      {mode === "text" ? (
        <Textarea value={textValue} onChange={(e) => onTextChange(e.target.value)} placeholder="Paste your copy here" className="rounded-2xl" />
      ) : (
        <div className="flex items-center gap-2 rounded-2xl border border-zinc-200 bg-white px-2 py-2">
          <label className="cursor-pointer rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-1 text-sm font-medium">
            Choose file
            <Input key={inputKey} type="file" accept={currentMode.accept} onChange={(e) => onSelect(e.target.files?.[0] || null)} className="sr-only" />
          </label>
          <div className="truncate text-sm text-zinc-500 flex-1">{file?.name || "No file chosen"}</div>
        </div>
      )}

      {mode === "text" ? (
        <div className="rounded-2xl border border-zinc-200/70 bg-zinc-50 p-4 min-h-32 text-sm text-zinc-600 line-clamp-6">{textValue || "Preview..."}</div>
      ) : file ? (
        <div className="rounded-2xl border border-zinc-200/70 bg-zinc-50 p-3">
          <div className="flex justify-between text-xs text-zinc-500 mb-2"><span>{file.name}</span><span>{fmtBytes(file.size)}</span></div>
          {file.type.startsWith("image/") && previewUrl && <img src={previewUrl} alt="preview" className="h-40 w-full rounded-xl object-cover" />}
          {file.type.startsWith("video/") && previewUrl && <video src={previewUrl} controls className="h-40 w-full rounded-xl object-cover" />}
          {file.type.startsWith("audio/") && previewUrl && <audio src={previewUrl} controls className="w-full" />}
        </div>
      ) : (
        <div className="flex items-center justify-center rounded-2xl border border-dashed border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-500 min-h-32">No file selected</div>
      )}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────

export default function HomePage() {
  const [mode, setMode] = useState("video");
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [textA, setTextA] = useState("");
  const [textB, setTextB] = useState("");
  
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [inputKey, setInputKey] = useState(0);
  const [toasts, setToasts] = useState([]);
  
  const [showQuickStart, setShowQuickStart] = useState(false);
  // null = no explicit choice yet: rail defaults open on xl, drawer closed below.
  const [sideRailOpen, setSideRailOpen] = useState(null);
  const [view, setView] = useState("upload");

  const isDesktopRail = useIsDesktopRail();
  const railVisible = (sideRailOpen ?? true) && isDesktopRail === true;
  const tipsSheetOpen = (sideRailOpen ?? false) && isDesktopRail === false;

  const previewA = useObjectUrl(fileA);
  const previewB = useObjectUrl(fileB);

  const pushToast = useCallback((t) => {
    const id = Date.now().toString();
    setToasts((c) => [...c, { id, ...t }]);
    setTimeout(() => setToasts((c) => c.filter((x) => x.id !== id)), TOAST_TIMEOUT_MS);
  }, []);

  useEffect(() => {
    if (!job?.job_id || job.status === "complete" || job.status === "failed") return;
    let cancelled = false;
    
    const intervalId = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${job.job_id}`, { cache: "no-store" });
        const data = await res.json();
        if (cancelled) return;
        
        if (!res.ok) throw new Error(data.error || "Status read failed.");
        setJob(data);
        
        if (data.status === "complete") {
          setResult(data.result);
          setBusy(false);
          setView("result");
          clearInterval(intervalId);
        } else if (data.status === "failed") {
          pushToast({ title: "Review failed", description: data.error || "Please try again.", tone: "error" });
          setBusy(false);
          clearInterval(intervalId);
        }
      } catch (ex) {
        if (!cancelled) {
          pushToast({ title: "Connection lost", description: "Please refresh and try again.", tone: "error" });
          setBusy(false);
          clearInterval(intervalId);
        }
      }
    }, 1500);
    return () => { cancelled = true; clearInterval(intervalId); };
  }, [job?.job_id, job?.status, pushToast]);

  const uploadA = mode === "text" ? (textA.trim() ? new File([textA], "A.txt", { type: "text/plain" }) : null) : fileA;
  const uploadB = mode === "text" ? (textB.trim() ? new File([textB], "B.txt", { type: "text/plain" }) : null) : fileB;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!uploadA) {
      pushToast({ title: "Version A is required.", description: "Add content for Version A." });
      return;
    }
    setBusy(true); setError(""); setResult(null); setJob(null); setView("upload");
    try {
      const fd = new FormData();
      fd.append("video_a", uploadA);
      if (uploadB) fd.append("video_b", uploadB);
      
      const res = await fetch(API_ENDPOINT, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not start review.");
      setJob(data);
    } catch (ex) {
      pushToast({ title: "Error starting review", description: ex.message });
      setBusy(false);
    }
  };

  const resetForm = () => {
    setMode("video"); setFileA(null); setFileB(null); setTextA(""); setTextB("");
    setJob(null); setResult(null); setBusy(false); setError(""); setView("upload");
    setInputKey((v) => v + 1);
  };

  const activePhase = busy ? (job?.message?.toLowerCase().includes("compar") ? 3 : job?.message?.toLowerCase().includes("score") ? 2 : 1) : (job?.status === "complete" ? 4 : 0);
  const progress = PROGRESS_PHASES[activePhase]?.percent || 0;

  const comparisonRows = useMemo(() => buildComparisonRows({ comparison: result?.comparison, stimulusA: result?.stimulus_a, stimulusB: result?.stimulus_b }), [result]);

  const chartData = useMemo(
    () => normalizeCurvesForChart([result?.stimulus_a?.curve || [], result?.stimulus_b?.curve || []]),
    [result],
  );

  return (
    <main className="app-bg flex h-[100svh] flex-col overflow-hidden text-zinc-950">
      <div className="pointer-events-none fixed top-16 right-4 z-40 grid w-[min(360px,calc(100vw-2rem))] gap-2">
        {toasts.map((t) => (
          <div key={t.id} className="pointer-events-auto rounded-2xl border border-zinc-200 bg-white p-3 shadow-sm">
            <div className="flex justify-between items-start gap-3">
              <div>
                <div className="text-sm font-semibold text-zinc-950">{t.title}</div>
                {t.description && <div className="text-xs text-zinc-600 mt-1">{t.description}</div>}
              </div>
              <button onClick={() => setToasts(c => c.filter(x => x.id !== t.id))} className="text-xs text-zinc-500 hover:text-zinc-950">Close</button>
            </div>
          </div>
        ))}
      </div>

      <header className="sticky top-0 z-30 h-14 border-b border-zinc-200/60 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex h-full max-w-[1640px] items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="brand-gradient flex size-8 items-center justify-center rounded-xl shadow-sm shadow-[hsl(243_75%_59%/0.35)]"><CommandIcon className="size-4 text-white" /></div>
            <div className="text-sm font-semibold tracking-tight">Compare <span className="gradient-text">Lab</span></div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="rounded-xl" onClick={() => setSideRailOpen(!(sideRailOpen ?? true))}>
              {(sideRailOpen ?? true) ? <PanelRightCloseIcon className="mr-2 size-4" /> : <PanelRightOpenIcon className="mr-2 size-4" />} Tips
            </Button>
            <Button variant="outline" size="sm" className="rounded-xl" aria-label="Quick start" onClick={() => setShowQuickStart(true)}>
              <InfoIcon className="size-4 sm:mr-2" /><span className="hidden sm:inline">Quick start</span>
            </Button>
          </div>
        </div>
      </header>

      <div className={`mx-auto grid h-[calc(100svh-3.5rem)] w-full max-w-[1640px] grid-cols-1 overflow-hidden ${railVisible ? "xl:grid-cols-[minmax(0,1fr)_360px]" : ""}`}>
        <section className="overflow-y-auto px-4 py-5 sm:px-6">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-5">
            <Card className="px-5 py-5">
              <div className="flex justify-between items-start gap-4">
                <SectionHeader tag="Workspace" title="Compare versions side by side" subtitle="Choose a format, add versions, and analyze the strongest option." />
                <div className="flex items-center gap-2">
                  {result && (
                    <Button variant="outline" size="sm" className="rounded-xl" onClick={() => setView(view === "result" ? "upload" : "result")}>
                      {view === "result" ? "Back to files" : "Open results"}
                    </Button>
                  )}
                  <Button variant="ghost" size="sm" className="rounded-xl" onClick={resetForm} disabled={busy}>Clear</Button>
                </div>
              </div>
            </Card>

            {view === "result" && result ? (
              <section className="grid gap-5">
                <Card className="p-5 grid gap-4">
                  <SectionHeader tag="Results" title={result.stimulus_b ? "Comparison Analysis" : "Single Version Review"} subtitle={result.disclaimer || "Decision support analysis."} />
                  
                  <div className={`grid gap-4 ${result.stimulus_b ? "lg:grid-cols-2" : ""}`}>
                    {[result.stimulus_a, result.stimulus_b].filter(Boolean).map((stim, i) => (
                      <div key={i} className="rounded-2xl border border-zinc-200/70 bg-zinc-50 p-4">
                        <div className="flex justify-between items-start mb-4">
                          <div className="flex items-center gap-3">
                            <StimulusCue label={stim.label} previewUrl={i === 0 ? previewA : previewB} mode={stim.modality} />
                            <div>
                              <p className="text-xs font-semibold text-zinc-500 uppercase">{versionLabel(stim.label)}</p>
                              <h3 className="text-sm font-semibold">{stim.asset?.name || stim.label}</h3>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="gradient-text text-4xl font-bold tabular-nums">{stim.scorecard?.overall_score}</div>
                            <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">{stim.scorecard?.overall_band}</div>
                          </div>
                        </div>
                        <p className="text-sm text-zinc-600 mb-4">{stim.scorecard?.summary}</p>
                        <div className="grid grid-cols-3 gap-2">
                          <div className="rounded-xl border bg-white p-2 text-center"><div className="text-xs text-zinc-500">Signal</div><div className="font-semibold text-sm">{stim.scorecard?.dominant_system?.label || "—"}</div></div>
                          <div className="rounded-xl border bg-white p-2 text-center"><div className="text-xs text-zinc-500">Best Moment</div><div className="font-semibold text-sm">{fmtSec(stim.scorecard?.peak_moment?.at_s)}</div></div>
                          <div className="rounded-xl border bg-white p-2 text-center"><div className="text-xs text-zinc-500">Pattern Mix</div><div className="font-semibold text-sm">{stim.scorecard?.laterality?.label || "—"}</div></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>

                {result.stimulus_b && comparisonRows.length > 0 && (
                  <Card className="p-5 grid gap-3">
                    <SectionHeader tag="Comparison" title="Where each version leads" />
                    <div className="rounded-2xl border border-zinc-200/70 bg-white overflow-hidden text-sm">
                       {comparisonRows.map((r, i) => (
                         <div key={i} className="border-b border-zinc-200/60 last:border-b-0">
                           <div className="grid grid-cols-[220px_1fr_1fr_96px] bg-zinc-50 font-medium">
                             <div className="p-3">{r.label}</div>
                             <div className={`p-3 ${r.winner === 'A' ? "bg-[hsl(var(--tenant-primary))]/10 text-[hsl(var(--tenant-primary))]" : ""}`}>{r.a}</div>
                             <div className={`p-3 ${r.winner === 'B' ? "bg-[hsl(var(--tenant-primary))]/10 text-[hsl(var(--tenant-primary))]" : ""}`}>{r.b}</div>
                             <div className="p-3 capitalize">{r.winner === "tie" ? "Even" : r.winner}</div>
                           </div>
                           <div className="p-3 text-xs text-zinc-600 bg-white">{r.insight}</div>
                         </div>
                       ))}
                    </div>
                  </Card>
                )}

                <div className="grid lg:grid-cols-2 gap-5">
                  <Card className="p-5">
                    <SectionHeader tag="Chart" title="Response over time" />
                    <div className="h-64 mt-4">
                      <ResultsChart data={chartData} showB={Boolean(result.stimulus_b)} />
                    </div>
                  </Card>

                  <Card className="p-5">
                    <SectionHeader tag="Notes" title="Key Observations" />
                    <div className="grid gap-2 mt-4">
                      {(result.observations || []).map((obs, i) => (
                        <div key={i} className="flex gap-3 rounded-xl border bg-zinc-50 p-3 text-sm text-zinc-700">
                          <SparklesIcon className="size-4 text-[hsl(var(--tenant-primary))] shrink-0 mt-0.5" />
                          <p>{obs}</p>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              </section>
            ) : (
              <form onSubmit={handleSubmit} className="grid gap-5 xl:grid-cols-[1fr_320px]">
                <div className="grid gap-4">
                  <Card className="p-4 flex flex-wrap gap-3 items-center justify-between">
                    <div>
                      <div className="text-xs font-semibold uppercase text-zinc-500">Format</div>
                      <div className="text-sm font-medium">Choose content type</div>
                    </div>
                    <div className="flex gap-2">
                      {STIMULUS_MODES.map((m) => (
                        <Button key={m.value} type="button" variant="outline" size="sm" onClick={() => { setMode(m.value); setFileA(null); setFileB(null); setTextA(""); setTextB(""); }} className={`rounded-full ${mode === m.value ? "brand-gradient border-transparent text-white shadow-sm" : ""}`}>
                          <m.icon className="mr-2 size-4" /> {m.label}
                        </Button>
                      ))}
                    </div>
                  </Card>

                  <Card className="grid lg:grid-cols-[1fr_72px_1fr] overflow-hidden">
                    <StimulusPanel inputKey={inputKey} title="Version A" description="Current version" mode={mode} file={fileA} textValue={textA} previewUrl={previewA} onSelect={setFileA} onTextChange={setTextA} />
                    <div className="hidden lg:flex items-center justify-center border-x bg-zinc-50">
                      <span className="brand-gradient flex size-10 items-center justify-center rounded-full text-xs font-bold text-white shadow-sm">VS</span>
                    </div>
                    <StimulusPanel inputKey={`${inputKey}-b`} title="Version B" description="Alternative" mode={mode} file={fileB} textValue={textB} previewUrl={previewB} optional onSelect={setFileB} onTextChange={setTextB} />
                  </Card>
                </div>

                <aside className="h-fit grid gap-4 rounded-3xl border bg-white p-5 xl:sticky xl:top-5">
                  <SectionHeader tag="Review" title="Ready to analyze" subtitle="Add files and start" />
                  
                  <div className="grid gap-2 text-sm">
                    <div className="flex justify-between p-2 rounded-xl border bg-zinc-50"><span className="flex items-center gap-2">{uploadA ? <CheckCircle2Icon className="size-4 text-[hsl(var(--tenant-primary))]" /> : <CircleDashedIcon className="size-4 text-zinc-400" />} Version A</span></div>
                    <div className="flex justify-between p-2 rounded-xl border bg-zinc-50"><span className="flex items-center gap-2">{uploadB ? <CheckCircle2Icon className="size-4 text-[hsl(var(--tenant-primary))]" /> : <CircleDashedIcon className="size-4 text-zinc-400" />} Version B</span></div>
                  </div>

                  <Button type="submit" disabled={busy || !uploadA} className="brand-gradient w-full rounded-2xl text-white shadow-lg shadow-[hsl(243_75%_59%/0.25)] hover:opacity-90 disabled:opacity-50">
                    {busy ? <LoaderCircleIcon className="mr-2 size-4 animate-spin" /> : <PlayIcon className="mr-2 size-4" />}
                    {uploadB ? "Compare versions" : "Review version"}
                  </Button>

                  {busy && (
                    <div className="grid gap-3 p-4 rounded-2xl border bg-zinc-50">
                      <div className="flex justify-between text-sm font-medium"><span>Progress</span><span>{progress}%</span></div>
                      <Progress value={progress} className="h-2 progress-shine" />
                      <div className="text-xs text-zinc-500">{PROGRESS_PHASES[activePhase]?.label || "Processing..."}</div>
                    </div>
                  )}
                </aside>
              </form>
            )}
          </div>
        </section>

        {railVisible && (
          <aside className="hidden border-l bg-zinc-50/70 p-5 xl:block overflow-y-auto">
            <SectionHeader tag="Tips" title="How to use this page" />
            <div className="mt-5">
              <TipsContent onOpenQuickStart={() => setShowQuickStart(true)} />
            </div>
          </aside>
        )}
      </div>

      <Sheet open={tipsSheetOpen} onOpenChange={(o) => setSideRailOpen(o)}>
        <SheetContent side="right" className="w-[300px] gap-4 border-l bg-zinc-50 sm:max-w-[300px]">
          <SheetHeader className="text-left">
            <SheetTitle className="text-sm font-semibold">How to use this page</SheetTitle>
            <SheetDescription>Quick tips for comparing versions.</SheetDescription>
          </SheetHeader>
          <div className="overflow-y-auto pb-4">
            <TipsContent onOpenQuickStart={() => { setSideRailOpen(false); setShowQuickStart(true); }} />
          </div>
        </SheetContent>
      </Sheet>
      <QuickStartDialog open={showQuickStart} onOpenChange={setShowQuickStart} />
    </main>
  );
}