import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';

const mock = new MockAdapter(axios, { delayResponse: 500 });
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const zones = [
    {
        id: 1,
        name: 'Kohima Village Area',
        district: 'Kohima',
        geom: 'POLYGON((94.09 25.65, 94.11 25.65, 94.11 25.67, 94.09 25.67, 94.09 25.65))',
        risk_level: 'HIGH',
        last_updated: new Date().toISOString(),
        recent_readings: [
            { id: 1, sensor_type: 'rainfall', value: 120, recorded_at: new Date().toISOString() },
            { id: 2, sensor_type: 'soil_moisture', value: 65, recorded_at: new Date().toISOString() },
        ],
        recent_predictions: [
            { id: 1, probability: 0.4, risk_level: 'MEDIUM', predicted_at: new Date(Date.now() - 3600000 * 3).toISOString() },
            { id: 2, probability: 0.6, risk_level: 'MEDIUM', predicted_at: new Date(Date.now() - 3600000 * 2).toISOString() },
            { id: 3, probability: 0.8, risk_level: 'HIGH', predicted_at: new Date(Date.now() - 3600000).toISOString() },
            { id: 4, probability: 0.9, risk_level: 'HIGH', predicted_at: new Date().toISOString() },
        ]
    },
    {
        id: 2,
        name: 'Jakhama',
        district: 'Kohima',
        geom: 'POLYGON((94.11 25.59, 94.13 25.59, 94.13 25.61, 94.11 25.61, 94.11 25.59))',
        risk_level: 'MEDIUM',
        last_updated: new Date().toISOString(),
        recent_readings: [],
        recent_predictions: []
    }
];

const alerts = [
    {
        id: 1,
        zone: 1,
        message: 'CRITICAL: Risk level for Kohima Village Area has escalated to HIGH. Probability: 0.90',
        channel: 'SMS/EMAIL',
        sent_at: new Date().toISOString()
    }
];

let reports = [
    {
        id: 1,
        reporter_name: 'John Doe',
        geom: { type: 'Point', coordinates: [94.1, 25.66] },
        description: 'Minor rockfall near the highway',
        status: 'PENDING',
        created_at: new Date().toISOString()
    }
];

mock.onGet(`${API_URL}/risk-zones/`).reply(200, zones);
mock.onGet(new RegExp(`${API_URL}/risk-zones/\\d+/`)).reply(config => {
    const id = parseInt(config.url.split('/').slice(-2, -1)[0]);
    const zone = zones.find(z => z.id === id);
    return [200, zone];
});

mock.onGet(`${API_URL}/alerts/`).reply(200, alerts);
mock.onGet(`${API_URL}/reports/`).reply(200, reports);

mock.onPost(`${API_URL}/reports/`).reply(config => {
    const data = JSON.parse(config.data);
    const newReport = {
        id: reports.length + 1,
        reporter_name: data.reporter_name,
        description: data.description,
        geom: { type: 'Point', coordinates: [94.1053, 25.6586] }, // dummy parsed
        status: 'PENDING',
        created_at: new Date().toISOString()
    };
    reports.unshift(newReport);
    return [201, newReport];
});

mock.onPatch(new RegExp(`${API_URL}/reports/\\d+/(verify|dismiss)/`)).reply(config => {
    const parts = config.url.split('/');
    const action = parts[parts.length - 2];
    const id = parseInt(parts[parts.length - 3]);
    const reportIndex = reports.findIndex(r => r.id === id);
    if (reportIndex !== -1) {
        reports[reportIndex].status = action === 'verify' ? 'VERIFIED' : 'DISMISSED';
    }
    return [200, { status: action }];
});

console.log('Mock API initialized');
