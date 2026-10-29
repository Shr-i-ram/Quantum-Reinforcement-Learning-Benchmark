import React from 'react';
import {
  Atom,
  LayoutDashboard,
  Gamepad2,
  Trophy,
  Swords,
  Cpu,
  Flame,
  BarChart3,
  BookOpen,
  Sparkles,
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  onSelectTab: (tabId: string) => void;
  onOpenTour: () => void;
}

const NAV_ITEMS = [
  { id: 'mission', label: 'Mission Control', icon: LayoutDashboard },
  { id: 'playground', label: 'Playground', icon: Gamepad2 },
  { id: 'challenges', label: 'Challenges', icon: Trophy },
  { id: 'duel', label: 'Head-to-Head', icon: Swords },
  { id: 'quantum', label: 'Quantum Lab', icon: Cpu },
  { id: 'training', label: 'Training Studio', icon: Flame },
  { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
  { id: 'insights', label: 'Research Insights', icon: BookOpen },
];

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  onOpenTour,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full bg-slate-950/90 backdrop-blur-md border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          {/* Logo & Platform Name */}
          <div
            onClick={() => onSelectTab('mission')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-600/30 border border-cyan-500/40 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition shadow-lg shadow-cyan-500/10">
              <Atom className="w-6 h-6 animate-[spin_12s_linear_infinite]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base text-white tracking-tight">
                  Quantum RL <span className="text-cyan-400">Arena</span>
                </span>
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-400 border border-cyan-800/50">
                  Research v2.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Classical vs Hybrid Quantum Reinforcement Learning
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                    isActive
                      ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Actions & Status */}
          <div className="flex items-center gap-3">
            {/* Guided Tour Presentation Button */}
            <button
              onClick={onOpenTour}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 shadow-md shadow-cyan-500/20 transition cursor-pointer"
            >
              <Sparkles className="w-3.5 h-3.5 fill-slate-950" />
              <span>Guided Tour</span>
            </button>

            {/* Live Indicator */}
            <div className="hidden xl:flex items-center gap-2 text-xs bg-slate-900/80 border border-slate-800 px-2.5 py-1.5 rounded-lg text-slate-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[11px] text-slate-300">FastAPI & PennyLane Live</span>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Scrollbar */}
        <div className="lg:hidden flex items-center gap-1 overflow-x-auto py-2 border-t border-slate-900">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium whitespace-nowrap transition ${
                  isActive
                    ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
