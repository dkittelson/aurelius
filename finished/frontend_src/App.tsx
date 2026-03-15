import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Dashboard from "./pages/Dashboard";
import ClustersPage from "./pages/ClustersPage";
import TopFlaggedPage from "./pages/TopFlaggedPage";
import { LayoutDashboard, AlertTriangle, List } from "lucide-react";

const queryClient = new QueryClient();

type Page = "dashboard" | "clusters" | "top-flagged";

const navItems: { id: Page; label: string; icon: React.ReactNode }[] = [
  { id: "dashboard",   label: "Dashboard",     icon: <LayoutDashboard size={16} /> },
  { id: "top-flagged", label: "Top Flagged",   icon: <List size={16} /> },
  { id: "clusters",    label: "AML Clusters",  icon: <AlertTriangle size={16} /> },
];

function AppShell() {
  const [page, setPage] = useState<Page>("dashboard");

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-52 flex-shrink-0 flex-col border-r border-zinc-800 bg-zinc-900">
        <div className="px-5 py-5">
          <h1 className="text-lg font-bold tracking-tight text-white">Aurelius</h1>
          <p className="text-xs text-zinc-500">AML Intelligence</p>
        </div>

        <nav className="flex flex-col gap-1 px-3">
          {navItems.map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                page === id
                  ? "bg-zinc-800 text-white"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              }`}
            >
              {icon}
              {label}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {page === "dashboard"   && <Dashboard />}
        {page === "top-flagged" && <TopFlaggedPage />}
        {page === "clusters"    && <ClustersPage />}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  );
}
