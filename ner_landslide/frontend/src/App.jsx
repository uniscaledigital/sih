import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import ZoneDetail from './pages/ZoneDetail';
import AlertsReports from './pages/AlertsReports';
import ReportForm from './pages/ReportForm';
import { Mountain, Settings, Bell, LayoutDashboard, Info, AlertTriangle } from 'lucide-react';

function Sidebar() {
  return (
    <aside className="fixed left-0 top-16 h-[calc(100vh-64px)] w-[280px] bg-surface-container-low border-r border-outline-variant flex-col py-md px-sm z-40 hidden md:flex">
      <div className="mb-lg px-md">
        <div className="text-label-md font-label-md text-on-surface-variant uppercase tracking-wider mb-2">Control Center</div>
        <div className="text-title-lg font-title-lg text-primary font-bold">Active Monitoring</div>
      </div>
      <nav className="flex-1 space-y-2">
        <Link to="/" className="flex items-center gap-4 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest transition-all duration-200 rounded-lg px-4 py-3">
          <LayoutDashboard size={20} />
          <span className="text-label-md font-label-md">Dashboard</span>
        </Link>
        <Link to="/alerts-reports" className="flex items-center gap-4 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest transition-all duration-200 rounded-lg px-4 py-3">
          <AlertTriangle size={20} />
          <span className="text-label-md font-label-md">Alerts & Reports</span>
        </Link>
        <Link to="/report" className="flex items-center gap-4 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest transition-all duration-200 rounded-lg px-4 py-3">
          <Info size={20} />
          <span className="text-label-md font-label-md">Citizen Report</span>
        </Link>
      </nav>
    </aside>
  );
}

function TopNavBar() {
  return (
    <nav className="flex justify-between items-center w-full px-lg h-16 fixed top-0 z-50 bg-background border-b border-outline-variant shadow-sm">
      <div className="flex items-center gap-md">
        <Mountain className="text-primary" size={32} />
        <span className="text-title-lg font-bold text-primary tracking-tight">NER Landslide System</span>
      </div>
      <div className="flex items-center gap-md">
        <button className="p-2 text-on-surface-variant hover:bg-surface-variant/10 rounded-full">
          <Bell size={24} />
        </button>
        <button className="p-2 text-on-surface-variant hover:bg-surface-variant/10 rounded-full">
          <Settings size={24} />
        </button>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="dark">
        <TopNavBar />
        <Sidebar />
        <main className="flex-1 ml-0 md:ml-[280px] mt-16 p-margin h-[calc(100vh-64px)] flex flex-col md:flex-row gap-lg overflow-hidden bg-background text-on-background">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/zone/:id" element={<ZoneDetail />} />
            <Route path="/alerts-reports" element={<AlertsReports />} />
            <Route path="/report" element={<ReportForm />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
