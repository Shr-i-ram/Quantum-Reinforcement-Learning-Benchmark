import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { GuidedTourModal } from './components/GuidedTourModal';
import { MissionControlView } from './views/MissionControlView';
import { PlaygroundView } from './views/PlaygroundView';
import { ChallengeArenaView } from './views/ChallengeArenaView';
import { HeadToHeadView } from './views/HeadToHeadView';
import { QuantumLabView } from './views/QuantumLabView';
import { TrainingStudioView } from './views/TrainingStudioView';
import { BenchmarkExplorerView } from './views/BenchmarkExplorerView';
import { ResearchInsightsView } from './views/ResearchInsightsView';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('mission');
  const [isTourOpen, setIsTourOpen] = useState<boolean>(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top Sticky Navigation */}
      <Navbar
        activeTab={activeTab}
        onSelectTab={(tabId) => setActiveTab(tabId)}
        onOpenTour={() => setIsTourOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-1">
        {activeTab === 'mission' && (
          <MissionControlView
            onNavigateTab={(tabId) => setActiveTab(tabId)}
            onOpenTour={() => setIsTourOpen(true)}
          />
        )}
        {activeTab === 'playground' && <PlaygroundView />}
        {activeTab === 'challenges' && <ChallengeArenaView />}
        {activeTab === 'duel' && <HeadToHeadView />}
        {activeTab === 'quantum' && <QuantumLabView />}
        {activeTab === 'training' && (
          <TrainingStudioView onNavigateTab={(tabId) => setActiveTab(tabId)} />
        )}
        {activeTab === 'benchmarks' && <BenchmarkExplorerView />}
        {activeTab === 'insights' && <ResearchInsightsView />}
      </main>

      {/* Guided Review Presentation Modal */}
      <GuidedTourModal
        isOpen={isTourOpen}
        onClose={() => setIsTourOpen(false)}
        onNavigateTab={(tabId) => {
          setActiveTab(tabId);
        }}
      />

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-2">
          <div>
            Quantum RL Arena • Research Project Platform v2.0
          </div>
          <div>
            Authors: Shriram Narayanan • Evelyn Maria Paulraj • Koppesh P.
          </div>
          <div className="text-slate-600">
            Powered by PyTorch, Gymnasium & PennyLane
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
