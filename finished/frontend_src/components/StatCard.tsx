interface Props {
  label: string;
  value: string | number;
  sub?: string;
  color?: "default" | "red" | "orange" | "yellow" | "green";
}

const colorMap: Record<string, string> = {
  default: "border-zinc-700 text-white",
  red: "border-red-500 text-red-400",
  orange: "border-orange-500 text-orange-400",
  yellow: "border-yellow-500 text-yellow-400",
  green: "border-green-500 text-green-400",
};

export default function StatCard({ label, value, sub, color = "default" }: Props) {
  return (
    <div className={`rounded-xl border bg-zinc-900 p-4 ${colorMap[color]}`}>
      <p className="text-xs uppercase tracking-widest text-zinc-400">{label}</p>
      <p className="mt-1 text-3xl font-bold">{value}</p>
      {sub && <p className="mt-1 text-xs text-zinc-500">{sub}</p>}
    </div>
  );
}
