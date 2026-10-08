import { useState, useEffect } from 'react';
import axios from 'axios';
import { AlertTriangle, Send, MapPin, WifiOff } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const initDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('LandslideReportsDB', 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('reportsQueue')) {
        db.createObjectStore('reportsQueue', { keyPath: 'id', autoIncrement: true });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const saveToQueue = async (payload) => {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('reportsQueue', 'readwrite');
    const store = tx.objectStore('reportsQueue');
    const request = store.add({ payload, timestamp: Date.now() });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

export const syncQueue = async () => {
  try {
    const db = await initDB();
    const tx = db.transaction('reportsQueue', 'readonly');
    const store = tx.objectStore('reportsQueue');
    const request = store.getAll();
    
    request.onsuccess = async () => {
      const queuedReports = request.result;
      if (queuedReports.length > 0) {
        console.log(`Attempting to sync ${queuedReports.length} queued reports...`);
        for (const record of queuedReports) {
          try {
            await axios.post(`${API_URL}/reports/`, record.payload);
            const deleteTx = db.transaction('reportsQueue', 'readwrite');
            deleteTx.objectStore('reportsQueue').delete(record.id);
            console.log(`Synced and removed report ${record.id} from queue.`);
          } catch (e) {
            console.error('Failed to sync queued report', e);
          }
        }
      }
    };
  } catch (e) {
    console.error('Error syncing queue', e);
  }
};

export default function ReportForm() {
  const [formData, setFormData] = useState({
    reporter_name: '',
    description: '',
  });
  const [location, setLocation] = useState(null);
  const [status, setStatus] = useState(''); // 'locating', 'submitting', 'success', 'error', 'queued'
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    setStatus('locating');
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
          setStatus('');
        },
        (error) => {
          console.error("Error getting location", error);
          setLocation({ lat: 25.6586, lng: 94.1053 });
          setStatus('');
        },
        { timeout: 5000, maximumAge: 60000 }
      );
    } else {
      setLocation({ lat: 25.6586, lng: 94.1053 });
      setStatus('');
    }

    const handleOnline = () => {
        setIsOnline(true);
        syncQueue();
    };
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    // Initial sync attempt if online on mount
    if (navigator.onLine) {
        syncQueue();
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!location) return;
    
    setStatus('submitting');
    const payload = {
      reporter_name: formData.reporter_name,
      description: formData.description,
      geom: `SRID=4326;POINT(${location.lng} ${location.lat})`
    };

    try {
      if (!isOnline) {
        throw new Error('Network offline');
      }
      await axios.post(`${API_URL}/reports/`, payload);
      setStatus('success');
      setFormData({ reporter_name: '', description: '' });
      setTimeout(() => setStatus(''), 3000);
    } catch (err) {
      if (!isOnline || err.code === 'ERR_NETWORK' || err.message === 'Network offline') {
        console.warn("Network error, queueing report locally.");
        try {
            await saveToQueue(payload);
            setStatus('queued');
            setFormData({ reporter_name: '', description: '' });
            setTimeout(() => setStatus(''), 5000);
        } catch (dbErr) {
            console.error("Failed to queue to IndexedDB", dbErr);
            setStatus('error');
        }
      } else {
        console.error(err);
        setStatus('error');
      }
    }
  };

  return (
    <div className="w-full max-w-md mx-auto my-auto slide-up">
      <div className="bg-surface-container-low border border-outline-variant rounded-2xl p-lg shadow-xl relative overflow-hidden">
        
        <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-error to-secondary-container"></div>
        
        <div className="text-center mb-lg pt-4">
          <div className="w-16 h-16 bg-error-container/20 rounded-full flex items-center justify-center mx-auto mb-4 border border-error/30">
            <AlertTriangle className="text-error" size={32} />
          </div>
          <h2 className="text-headline-md font-bold text-on-surface">Report Hazard</h2>
          <p className="text-body-md text-on-surface-variant mt-2">Help authorities by reporting landslides or suspicious ground movement.</p>
          {!isOnline && (
            <div className="mt-2 flex items-center justify-center gap-2 text-error">
                <WifiOff size={16} />
                <span className="text-label-md">You are offline. Reports will be queued.</span>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          
          <div className="bg-surface border border-outline-variant rounded-xl p-md flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-primary-container/20 flex items-center justify-center text-primary">
              <MapPin size={20} />
            </div>
            <div>
              <p className="text-label-md text-on-surface-variant uppercase tracking-wider">Current Location</p>
              <p className="text-body-md font-mono-data text-on-surface">
                {status === 'locating' ? 'Acquiring GPS...' : location ? `${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}` : 'Location unavailable'}
              </p>
            </div>
          </div>

          <div>
            <label className="block text-label-md font-bold text-on-surface-variant mb-2 ml-1">Your Name</label>
            <input 
              type="text" 
              name="reporter_name"
              required
              value={formData.reporter_name}
              onChange={handleChange}
              className="w-full bg-surface border border-outline-variant rounded-xl px-4 py-3 text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              placeholder="Enter your full name"
            />
          </div>

          <div>
            <label className="block text-label-md font-bold text-on-surface-variant mb-2 ml-1">Description</label>
            <textarea 
              name="description"
              required
              rows="4"
              value={formData.description}
              onChange={handleChange}
              className="w-full bg-surface border border-outline-variant rounded-xl px-4 py-3 text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all resize-none"
              placeholder="Describe what you see (e.g., deep cracks on road, falling rocks)..."
            ></textarea>
          </div>

          <button 
            type="submit" 
            disabled={status === 'locating' || status === 'submitting'}
            className="w-full bg-error hover:bg-error/90 text-on-error font-bold text-title-lg py-4 rounded-xl transition-all flex justify-center items-center gap-2 active:scale-95 disabled:opacity-50"
          >
            {status === 'submitting' ? 'Submitting...' : (
              <>
                <Send size={20} /> Submit Report
              </>
            )}
          </button>
          
          {status === 'success' && (
            <p className="text-primary text-center font-bold mt-4 slide-up">Report submitted successfully!</p>
          )}
          {status === 'error' && (
            <p className="text-error text-center font-bold mt-4">Failed to submit. Please try again.</p>
          )}
          {status === 'queued' && (
            <div className="mt-4 p-3 bg-secondary-container text-on-secondary-container rounded-lg slide-up flex items-center justify-center gap-2">
                <WifiOff size={18} />
                <p className="font-bold text-center">Queued — will send when back online!</p>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
