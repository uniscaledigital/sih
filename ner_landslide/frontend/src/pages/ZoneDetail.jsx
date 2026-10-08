import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { ArrowLeft } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export default function ZoneDetail() {
  const { id } = useParams();
  const [zone, setZone] = useState(null);

  useEffect(() => {
    fetchZone();
  }, [id]);

  const fetchZone = async () => {
    try {
      const res = await axios.get(`${API_URL}/risk-zones/${id}/`);
      setZone(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  if (!zone) return <div className="p-lg">Loading...</div>;

  const chartData = [...zone.recent_predictions].reverse().map(p => ({
    time: new Date(p.predicted_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
    probability: p.probability * 100
  }));

  const ml = zone.ml_prediction;
  const envWarning = ml?.environmental_warning;

  // Extract latest live readings
  const latestRainfall = zone.recent_readings.find(r => r.sensor_type === 'rainfall');
  const latestSoil = zone.recent_readings.find(r => r.sensor_type === 'soil_moisture');
  const latestSeismic = zone.recent_readings.find(r => r.sensor_type === 'seismic_activity');

  return (
    <div className="w-full flex flex-col gap-lg overflow-y-auto pb-20">
      <div className="flex items-center gap-4">
        <Link to="/" className="p-2 bg-surface-variant text-on-surface hover:bg-surface-container-highest rounded-full">
            <ArrowLeft size={20} />
        </Link>
        <h2 className="text-display-lg text-primary">{zone.name}</h2>
        <span className={`px-4 py-1 rounded-full text-label-md font-bold ${
            zone.risk_level === 'VERY HIGH' ? 'bg-[#b3261e] text-white' :
            zone.risk_level === 'HIGH' ? 'bg-error text-on-error' :
            zone.risk_level === 'MODERATE' ? 'bg-[#ee9800] text-black' :
            zone.risk_level === 'MEDIUM' ? 'bg-[#ee9800] text-black' :
            'bg-primary-container text-on-primary-container'
        }`}>
            {zone.risk_level} RISK
        </span>
      </div>

      {ml && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-surface-container p-md rounded-xl border border-outline-variant widget-glow">
            <p className="text-label-md text-on-surface-variant uppercase">Integrated Risk Score</p>
            <p className="text-headline-lg font-bold text-primary">{(ml.risk_score * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-surface-container p-md rounded-xl border border-outline-variant widget-glow">
            <p className="text-label-md text-on-surface-variant uppercase">Susceptibility</p>
            <p className="text-headline-lg font-bold text-primary">{(ml.susceptibility_probability * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-surface-container p-md rounded-xl border border-outline-variant widget-glow">
            <p className="text-label-md text-on-surface-variant uppercase">Dynamic Trigger</p>
            <p className="text-headline-lg font-bold text-primary">{(ml.trigger_probability * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}

      {envWarning && (
        <div className={`p-lg rounded-xl border widget-glow ${
          envWarning.status === 'CRITICAL' ? 'bg-[#b3261e]/20 border-[#b3261e]' :
          envWarning.status === 'WARNING' ? 'bg-error/20 border-error' :
          envWarning.status === 'WATCH' ? 'bg-[#ee9800]/20 border-[#ee9800]' :
          'bg-surface-container border-outline-variant'
        }`}>
          <h3 className="text-title-lg font-bold mb-2 uppercase flex items-center gap-2">
            Early Warning Status: {envWarning.status}
          </h3>
          <p className="text-body-lg font-medium">{envWarning.reason}</p>
          <p className="text-label-sm text-on-surface-variant mt-2">
            Updated: {new Date(envWarning.timestamp).toLocaleString()}
          </p>
        </div>
      )}

      <div className="bg-surface-container border border-outline-variant p-lg rounded-xl widget-glow">
        <h3 className="text-title-lg font-bold mb-md uppercase text-primary">Live Environmental Conditions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-md bg-surface-container-highest rounded border border-outline-variant">
            <p className="text-label-md text-on-surface-variant uppercase flex items-center gap-2">🌧 Rainfall</p>
            {latestRainfall ? (
              <>
                <p className="text-headline-lg text-primary">{latestRainfall.value.toFixed(2)} mm/h</p>
                <p className="text-label-md mt-2 text-outline">{new Date(latestRainfall.recorded_at).toLocaleTimeString()}</p>
              </>
            ) : <p className="text-body-md text-on-surface-variant mt-2">Data unavailable</p>}
          </div>
          <div className="p-md bg-surface-container-highest rounded border border-outline-variant">
            <p className="text-label-md text-on-surface-variant uppercase flex items-center gap-2">💧 Soil Moisture</p>
            {latestSoil ? (
              <>
                <p className="text-headline-lg text-primary">{latestSoil.value.toFixed(1)}%</p>
                <p className="text-label-md mt-2 text-outline">{new Date(latestSoil.recorded_at).toLocaleTimeString()}</p>
              </>
            ) : <p className="text-body-md text-on-surface-variant mt-2">Data unavailable</p>}
          </div>
          <div className="p-md bg-surface-container-highest rounded border border-outline-variant">
            <p className="text-label-md text-on-surface-variant uppercase flex items-center gap-2">🌎 Seismic Activity</p>
            {latestSeismic ? (
              <>
                <p className="text-headline-lg text-primary">{latestSeismic.value} events</p>
                <p className="text-label-md mt-2 text-outline">Past 7 days</p>
              </>
            ) : <p className="text-body-md text-on-surface-variant mt-2">Data unavailable</p>}
          </div>
        </div>
      </div>

        <div className="bg-surface-container border border-outline-variant p-lg rounded-xl widget-glow h-[400px]">
          <h3 className="text-title-lg font-bold mb-md">Risk Probability Trend</h3>
          {chartData.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#3c4a42" />
                <XAxis dataKey="time" stroke="#c0c9c1" tick={{fontSize: 12}} />
                <YAxis stroke="#c0c9c1" tick={{fontSize: 12}} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{backgroundColor: '#191c1a', borderColor: '#3c4a42'}}
                  itemStyle={{color: '#4edea3'}}
                />
                <Line type="monotone" dataKey="probability" stroke="#4edea3" strokeWidth={3} dot={{r: 4, fill: '#4edea3'}} activeDot={{r: 6}} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <p className="text-on-surface-variant font-medium">Insufficient historical predictions for trend.</p>
            </div>
          )}
        </div>

      <div className="bg-surface-container border border-outline-variant p-lg rounded-xl widget-glow">
        <h3 className="text-title-lg font-bold mb-1">AI Risk Explanation</h3>
        <p className="text-label-md text-on-surface-variant mb-md">Model Feature Importance</p>
        
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-end mb-1">
              <span className="font-bold text-on-surface">1. Elevation</span>
              <span className="text-sm font-bold text-primary">{zone.terrain ? `${Math.round(zone.terrain.elevation)} m` : 'Data unavailable'}</span>
            </div>
            <div className="w-full bg-surface-container-highest h-2 rounded overflow-hidden">
              <div className="bg-[#b3261e] h-full" style={{width: '22.9%'}}></div>
            </div>
            <p className="text-xs text-on-surface-variant mt-1">High static model importance (22.9%). Increases susceptibility.</p>
          </div>
          <div>
            <div className="flex justify-between items-end mb-1">
              <span className="font-bold text-on-surface">2. 30-Day Cumulative Rainfall</span>
              <span className="text-sm text-on-surface-variant">Data automatically computed (rolling)</span>
            </div>
            <div className="w-full bg-surface-container-highest h-2 rounded overflow-hidden">
              <div className="bg-[#4edea3] h-full" style={{width: '17.7%'}}></div>
            </div>
            <p className="text-xs text-on-surface-variant mt-1">Highest dynamic trigger importance (17.7%).</p>
          </div>
          <div>
            <div className="flex justify-between items-end mb-1">
              <span className="font-bold text-on-surface">3. Slope</span>
              <span className="text-sm font-bold text-primary">{zone.terrain ? `${zone.terrain.slope.toFixed(1)}°` : 'Data unavailable'}</span>
            </div>
            <div className="w-full bg-surface-container-highest h-2 rounded overflow-hidden">
              <div className="bg-[#ee9800] h-full" style={{width: '13.7%'}}></div>
            </div>
            <p className="text-xs text-on-surface-variant mt-1">High static model importance (13.7%).</p>
          </div>
          <div>
            <div className="flex justify-between items-end mb-1">
              <span className="font-bold text-on-surface">4. Current Soil Moisture (0-1cm)</span>
              <span className="text-sm font-bold text-primary">{latestSoil ? `${latestSoil.value.toFixed(1)}%` : 'Data unavailable'}</span>
            </div>
            <div className="w-full bg-surface-container-highest h-2 rounded overflow-hidden">
              <div className="bg-primary h-full" style={{width: '8.0%'}}></div>
            </div>
            <p className="text-xs text-on-surface-variant mt-1">Current environmental condition.</p>
          </div>
          <div>
            <div className="flex justify-between items-end mb-1">
              <span className="font-bold text-on-surface">5. Current Rainfall</span>
              <span className="text-sm font-bold text-primary">{latestRainfall ? `${latestRainfall.value.toFixed(1)} mm/h` : 'Data unavailable'}</span>
            </div>
            <div className="w-full bg-surface-container-highest h-2 rounded overflow-hidden">
              <div className="bg-primary h-full" style={{width: '6.0%'}}></div>
            </div>
            <p className="text-xs text-on-surface-variant mt-1">Current environmental condition.</p>
          </div>
        </div>
      </div>

      <div className="bg-surface-container border border-outline-variant p-lg rounded-xl widget-glow">
        <h3 className="text-title-lg font-bold mb-md">Recent Sensor Readings (Raw)</h3>
        {zone.recent_readings && zone.recent_readings.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {zone.recent_readings.slice(0, 4).map(reading => (
              <div key={reading.id} className="p-sm bg-surface-container-highest rounded border border-outline-variant">
                <p className="text-label-sm text-on-surface-variant uppercase">{reading.sensor_type.replace('_', ' ')}</p>
                <p className="text-title-md text-on-surface">{reading.value.toFixed(2)}</p>
                <p className="text-label-sm mt-1 text-outline">{new Date(reading.recorded_at).toLocaleTimeString()}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-on-surface-variant text-sm">No recent raw sensor readings available.</p>
        )}
      </div>

    </div>
  );
}
