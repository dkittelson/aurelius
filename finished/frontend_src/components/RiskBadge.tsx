type Risk = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

const styles: Record<Risk, string> = {
  CRITICAL: "bg-red-900 text-red-300 border border-red-700",
  HIGH:     "bg-orange-900 text-orange-300 border border-orange-700",
  MEDIUM:   "bg-yellow-900 text-yellow-300 border border-yellow-700",
  LOW:      "bg-zinc-800 text-zinc-300 border border-zinc-600",
};

export default function RiskBadge({ level }: { level: Risk | string }) {
  const l = level as Risk;
  return (
    <span className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${styles[l] ?? styles.LOW}`}>
      {level}
    </span>
  );
}
