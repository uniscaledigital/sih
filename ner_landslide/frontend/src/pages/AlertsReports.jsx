import { useEffect, useState } from 'react';
import axios from 'axios';
import { Check, X } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export default function AlertsReports() {
  const [activeTab, setActiveTab] = useState('alerts');
  const [alerts, setAlerts] = useState([]);
  const [reports, setReports] = useState([]);

  useEffect(() => {
    fetchAlerts();
    fetchReports();
  }, []);

  const fetchAlerts = async () => {
    try {
      const res = await axios.get(`${API_URL}/alerts/`);
      setAlerts(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchReports = async () => {
    try {
      const res = await axios.get(`${API_URL}/reports/`);
      setReports(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const updateReportStatus = async (id, action) => {
    try {
      await axios.patch(`${API_URL}/reports/${id}/${action}/`);
      fetchReports();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="w-full h-full flex flex-col gap-lg bg-surface-container-lowest p-lg rounded-xl border border-outline-variant shadow-lg overflow-hidden">
      <div className="flex gap-4 border-b border-outline-variant pb-2">
        <button 
          className={`text-title-lg font-bold px-4 py-2 rounded-t-lg transition-colors ${activeTab === 'alerts' ? 'bg-primary-container text-on-primary-container' : 'text-on-surface-variant hover:text-on-surface'}`}
          onClick={() => setActiveTab('alerts')}
        >
          System Alerts
        </button>
        <button 
          className={`text-title-lg font-bold px-4 py-2 rounded-t-lg transition-colors ${activeTab === 'reports' ? 'bg-primary-container text-on-primary-container' : 'text-on-surface-variant hover:text-on-surface'}`}
          onClick={() => setActiveTab('reports')}
        >
          Citizen Reports
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {activeTab === 'alerts' && (
          <div className="space-y-4">
            {alerts.map(alert => (
              <div key={alert.id} className="p-md bg-error-container text-on-error-container rounded-xl border border-error shadow-sm">
                <div className="flex justify-between items-start mb-2">
                    <span className="font-bold text-title-lg">Critical Alert</span>
                    <span className="text-label-md bg-background/50 px-2 py-1 rounded">{new Date(alert.sent_at).toLocaleString()}</span>
                </div>
                <p className="text-body-lg">{alert.message}</p>
                <p className="text-label-md mt-4 text-on-error-container/80">Channel: {alert.channel}</p>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'reports' && (
          <div className="space-y-4">
            {reports.map(report => (
              <div key={report.id} className="p-md bg-surface-container rounded-xl border border-outline-variant shadow-sm flex flex-col md:flex-row justify-between gap-md">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-bold text-title-lg text-primary">{report.reporter_name}</span>
                    <span className={`text-label-md px-2 py-1 rounded-full ${
                        report.status === 'VERIFIED' ? 'bg-primary-container text-on-primary-container' :
                        report.status === 'DISMISSED' ? 'bg-error-container text-on-error-container' :
                        'bg-secondary-container text-on-secondary-container'
                    }`}>{report.status}</span>
                  </div>
                  <p className="text-body-lg text-on-surface mb-2">{report.description}</p>
                  <p className="text-label-md text-on-surface-variant">{new Date(report.created_at).toLocaleString()}</p>
                  <p className="text-label-md text-on-surface-variant mt-1">
                     Location: {report.geom ? report.geom.replace('SRID=4326;POINT', '').replace('(', '').replace(')', '').trim() : 'N/A'}
                  </p>
                </div>
                
                {report.status === 'PENDING' && (
                  <div className="flex gap-2 items-center">
                    <button 
                        onClick={() => updateReportStatus(report.id, 'verify')}
                        className="p-2 bg-primary-container text-on-primary-container rounded hover:bg-primary-fixed-dim transition-colors flex items-center justify-center"
                        title="Verify"
                    >
                      <Check size={20} />
                    </button>
                    <button 
                        onClick={() => updateReportStatus(report.id, 'dismiss')}
                        className="p-2 bg-error-container text-on-error-container rounded hover:bg-error transition-colors flex items-center justify-center"
                        title="Dismiss"
                    >
                      <X size={20} />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
