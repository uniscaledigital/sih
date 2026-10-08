import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Polygon, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { Map } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export default function Dashboard() {
  const [zones, setZones] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [boundaries, setBoundaries] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchZones(), fetchAlerts(), fetchBoundaries()]).finally(() => setIsLoading(false));
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchZones = async () => {
    try {
      const res = await axios.get(`${API_URL}/risk-zones/`);
      setZones(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchBoundaries = async () => {
    try {
      const res = await axios.get(`${API_URL}/boundaries/`);
      setBoundaries(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await axios.get(`${API_URL}/alerts/`);
      setAlerts(res.data.slice(0, 5)); // top 5
    } catch (err) {
      console.error(err);
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'VERY HIGH': return '#b3261e'; // deep red
      case 'HIGH': return '#ffb4ab'; // error
      case 'MODERATE': return '#ee9800'; // warning
      case 'MEDIUM': return '#ee9800'; // legacy fallback
      case 'LOW': return '#4edea3'; // primary
      default: return '#86948a'; // outline
    }
  };

  const parseWKTToCoordinates = (wkt) => {
    try {
      const match = wkt.match(/\(\(([^)]+)\)\)/);
      if (!match) return [];
      const coordStrings = match[1].split(',');
      return coordStrings.map(coord => {
        const [lon, lat] = coord.trim().split(' ').map(Number);
        return [lat, lon];
      });
    } catch (e) {
      return [];
    }
  };

  const parseMultiWKTToCoordinatesList = (wkt) => {
    try {
      const rings = [...wkt.matchAll(/\(\(([^)]+)\)\)/g)];
      return rings.map(match => {
        const coordStrings = match[1].split(',');
        return coordStrings.map(coord => {
          const [lon, lat] = coord.trim().split(' ').map(Number);
          return [lat, lon];
        });
      });
    } catch (e) {
      return [];
    }
  };

  return (
    <div className="flex w-full h-full gap-lg">
      <div className="flex-1 relative rounded-xl overflow-hidden border border-outline-variant shadow-lg bg-surface-container-lowest">
        {isLoading ? (
          <div className="w-full h-full bg-surface-container animate-pulse flex items-center justify-center">
            <p className="text-on-surface-variant font-bold text-title-lg">Loading Map Data...</p>
          </div>
        ) : zones.length === 0 ? (
          <div className="w-full h-full flex flex-col items-center justify-center bg-surface-container-lowest p-lg text-center">
            <div className="w-24 h-24 bg-primary-container/20 text-primary rounded-full flex items-center justify-center mb-4 border border-primary/30">
              <Map size={48} />
            </div>
            <h2 className="text-headline-md font-bold text-on-surface">No Risk Zones Configured</h2>
            <p className="text-body-lg text-on-surface-variant mt-2 max-w-md">The system is active, but no geographic risk zones have been loaded into the database yet.</p>
          </div>
        ) : (
          <MapContainer center={[25.6, 94.1]} zoom={11} className="h-full w-full">
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            {boundaries.map((boundary) => {
              const multiPositions = parseMultiWKTToCoordinatesList(boundary.geom);
              return multiPositions.length > 0 ? (
                <Polygon 
                  key={`boundary-${boundary.id}`} 
                  positions={multiPositions} 
                  pathOptions={{ 
                      color: '#4a5950', // dark grey outline
                      fillColor: 'transparent',
                      weight: 2,
                      dashArray: '5, 5'
                  }}
                />
              ) : null;
            })}
            {zones.map((zone) => {
              const positions = parseWKTToCoordinates(zone.geom);
              return positions.length > 0 ? (
                <Polygon 
                  key={zone.id} 
                  positions={positions} 
                  pathOptions={{ 
                      color: getRiskColor(zone.risk_level),
                      fillColor: getRiskColor(zone.risk_level),
                      fillOpacity: 0.4
                  }}
                >
                  <Popup className="ml-popup">
                    <div className="text-[#111827] min-w-[300px]">
                        <h3 className="font-bold text-lg mb-1">{zone.name}</h3>
                        {zone.ml_prediction ? (
                          <>
                            <div className="mt-2 text-sm">
                              <p className="text-xs font-bold text-[#059669] mb-2 uppercase tracking-wide">Prototype ML Prediction</p>
                              <p className="mb-1 text-[#374151]"><strong>Risk Level:</strong> <span className="text-[#111827] font-semibold">{zone.risk_level}</span></p>
                              <p className="mb-1 text-[#374151]"><strong>Integrated Risk Score:</strong> <span className="text-[#111827] font-semibold">{(zone.ml_prediction.risk_score * 100).toFixed(1)}%</span></p>
                              <p className="mb-1 text-[#374151]"><strong>Susceptibility:</strong> <span className="text-[#111827] font-semibold">{(zone.ml_prediction.susceptibility_probability * 100).toFixed(1)}%</span></p>
                              <p className="mb-1 text-[#374151]"><strong>Dynamic Trigger:</strong> <span className="text-[#111827] font-semibold">{(zone.ml_prediction.trigger_probability * 100).toFixed(1)}%</span></p>
                              
                              {zone.ml_prediction.environmental_warning && (
                                <div className="mt-3 border-t border-gray-300 pt-2">
                                  <p className="text-xs font-bold text-[#ee9800] mb-1 uppercase tracking-wide">Environmental Warning</p>
                                  <p className="mb-1 text-[#374151]"><strong>Status:</strong> <span className="font-bold text-[#111827]">{zone.ml_prediction.environmental_warning.status}</span></p>
                                  <p className="mb-1 text-xs text-[#111827] font-medium">{zone.ml_prediction.environmental_warning.reason}</p>
                                  <p className="text-[10px] text-[#4b5563] mt-1">Updated: {new Date(zone.ml_prediction.environmental_warning.timestamp).toLocaleString()}</p>
                                </div>
                              )}

                              <div className="mt-3 border-t border-gray-300 pt-2">
                                <p className="text-xs font-bold text-[#059669] mb-2 uppercase tracking-wide">Model Feature Importance</p>
                                <p className="text-xs font-bold text-[#111827] mb-1">Top Contributing Factors:</p>
                                <div className="space-y-2 mt-2">
                                  <div>
                                    <p className="text-[11px] font-bold text-[#111827] uppercase tracking-wider">1. ELEVATION</p>
                                    <p className="text-[11px] font-bold text-[#111827]">{zone.terrain ? `${Math.round(zone.terrain.elevation)} m` : 'Data unavailable'}</p>
                                    <p className="text-[10px] text-[#374151]">High static model importance (22.9%)</p>
                                  </div>
                                  <div>
                                    <p className="text-[11px] font-bold text-[#111827] uppercase tracking-wider">2. 30-DAY RAINFALL</p>
                                    <p className="text-[11px] font-bold text-[#111827]">Dynamic trigger</p>
                                    <p className="text-[10px] text-[#374151]">Highest dynamic importance (17.7%)</p>
                                  </div>
                                  <div>
                                    <p className="text-[11px] font-bold text-[#111827] uppercase tracking-wider">3. SLOPE</p>
                                    <p className="text-[11px] font-bold text-[#111827]">{zone.terrain ? `${zone.terrain.slope.toFixed(1)}°` : 'Data unavailable'}</p>
                                    <p className="text-[10px] text-[#374151]">High static model importance (13.7%)</p>
                                  </div>
                                </div>
                              </div>
                              
                              <p className="text-xs text-[#4b5563] mt-3 border-t border-gray-300 pt-2">
                                Lat: {zone.ml_prediction.latitude?.toFixed(4)}, Lon: {zone.ml_prediction.longitude?.toFixed(4)}
                              </p>
                            </div>
                            <Link to={`/zone/${zone.id}`} className="text-[#059669] hover:text-[#047857] hover:underline mt-3 block text-sm font-bold text-center w-full bg-gray-100 py-2 rounded border border-gray-200">View Full Details</Link>
                          </>
                        ) : (
                          <div className="mt-2 text-[#111827]">
                            <p>Risk: {zone.risk_level}</p>
                            <Link to={`/zone/${zone.id}`} className="text-[#059669] hover:underline mt-2 block font-bold">View Details</Link>
                          </div>
                        )}
                    </div>
                  </Popup>
                </Polygon>
              ) : null;
            })}
          </MapContainer>
        )}
      </div>

      <div className="w-80 flex flex-col gap-lg overflow-y-auto">
        
        {/* EARLY WARNING STATUS */}
        <div className="bg-surface-container border border-outline-variant rounded-xl p-md widget-glow">
          <h3 className="text-title-lg font-bold text-primary mb-sm uppercase tracking-wide">Early Warning Status</h3>
          {isLoading ? (
            <div className="h-32 bg-surface-container-highest rounded border border-outline-variant animate-pulse"></div>
          ) : (() => {
            const monitoredZones = zones.filter(z => z.ml_prediction?.environmental_warning);
            const critCount = monitoredZones.filter(z => z.ml_prediction.environmental_warning.status === 'CRITICAL').length;
            const warnCount = monitoredZones.filter(z => z.ml_prediction.environmental_warning.status === 'WARNING').length;
            const watchCount = monitoredZones.filter(z => z.ml_prediction.environmental_warning.status === 'WATCH').length;
            const normCount = monitoredZones.filter(z => z.ml_prediction.environmental_warning.status === 'NORMAL').length;

            return (
              <div className="space-y-3">
                <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#b3261e]"></div>CRITICAL</span><span className="font-bold">{critCount}</span></div>
                <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-error"></div>WARNING</span><span className="font-bold">{warnCount}</span></div>
                <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#ee9800]"></div>WATCH</span><span className="font-bold">{watchCount}</span></div>
                <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#4edea3]"></div>NORMAL</span><span className="font-bold">{normCount}</span></div>
                
                <div className="pt-2 border-t border-outline-variant text-xs text-on-surface-variant text-right italic">
                  Live monitoring coverage: {monitoredZones.length} zones
                </div>
              </div>
            );
          })()}
        </div>

        {/* RISK SUMMARY */}
        <div className="bg-surface-container border border-outline-variant rounded-xl p-md widget-glow">
          <h3 className="text-title-lg font-bold text-primary mb-sm uppercase tracking-wide">ML Risk Distribution</h3>
          {isLoading ? (
            <div className="h-32 bg-surface-container-highest rounded border border-outline-variant animate-pulse"></div>
          ) : (
            <div className="space-y-3">
              <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#b3261e]"></div>VERY HIGH</span><span className="font-bold">{zones.filter(z => z.risk_level === 'VERY HIGH').length}</span></div>
              <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#ffb4ab]"></div>HIGH</span><span className="font-bold">{zones.filter(z => z.risk_level === 'HIGH').length}</span></div>
              <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#ee9800]"></div>MODERATE</span><span className="font-bold">{zones.filter(z => z.risk_level === 'MODERATE').length}</span></div>
              <div className="flex justify-between items-center"><span className="text-on-surface flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#4edea3]"></div>LOW</span><span className="font-bold">{zones.filter(z => z.risk_level === 'LOW').length}</span></div>
            </div>
          )}
        </div>

        {/* EARLY WARNING CENTER FEED */}
        <div className="bg-surface-container border border-outline-variant rounded-xl p-md widget-glow">
          <h3 className="text-title-lg font-bold text-primary mb-sm uppercase tracking-wide">Early Warning Center</h3>
          <div className="space-y-3">
            {isLoading ? (
              [1, 2, 3].map(i => (
                <div key={i} className="h-24 bg-surface-container-highest rounded border border-outline-variant animate-pulse"></div>
              ))
            ) : (() => {
              const activeWarnings = zones
                .filter(z => z.ml_prediction?.environmental_warning && z.ml_prediction.environmental_warning.status !== 'NORMAL')
                .map(z => ({ zone: z, warning: z.ml_prediction.environmental_warning }))
                .sort((a, b) => new Date(b.warning.timestamp) - new Date(a.warning.timestamp));

              if (activeWarnings.length === 0) {
                return (
                  <div className="p-4 bg-primary-container/20 text-on-surface border border-primary/30 rounded-lg text-center">
                    <p className="text-title-md font-bold text-primary mb-1">✓ NO ACTIVE ALERTS</p>
                    <p className="text-sm">System is monitoring environmental conditions continuously.</p>
                  </div>
                );
              }

              return activeWarnings.slice(0, 5).map(({ zone, warning }) => (
                <div key={zone.id} className={`p-sm rounded-lg border ${
                  warning.status === 'CRITICAL' ? 'bg-[#b3261e]/20 border-[#b3261e]' :
                  warning.status === 'WARNING' ? 'bg-error/20 border-error' :
                  'bg-[#ee9800]/20 border-[#ee9800]'
                }`}>
                  <div className="flex justify-between items-start mb-1">
                    <p className="text-label-md font-bold tracking-wide uppercase">{warning.status}</p>
                    <p className="text-[10px] text-on-surface-variant">{new Date(warning.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</p>
                  </div>
                  <Link to={`/zone/${zone.id}`} className="text-title-md font-bold hover:underline block mb-1">{zone.name}</Link>
                  <p className="text-sm font-medium text-on-surface mb-2">{warning.reason}</p>
                  <p className="text-xs text-on-surface-variant">ML Risk: {zone.risk_level}</p>
                </div>
              ));
            })()}
          </div>
        </div>
      </div>
    </div>
  );
}
