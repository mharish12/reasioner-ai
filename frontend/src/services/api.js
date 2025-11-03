import axios from 'axios';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Agents API
export const agentsAPI = {
  getAll: () => api.get('/api/agents/'),
  getById: (id) => api.get(`/api/agents/${id}`),
  create: (data) => api.post('/api/agents/', data),
  delete: (id) => api.delete(`/api/agents/${id}`),
};

// Models API
export const modelsAPI = {
  getAll: (agentId = null) => {
    const params = agentId ? { agent_id: agentId } : {};
    return api.get('/api/models/', { params });
  },
  getById: (id) => api.get(`/api/models/${id}`),
  train: (formData) =>
    api.post('/api/train/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
};

// Query API
export const queryAPI = {
  query: (data) => api.post('/api/query/', data),
};

// Unlearn API
export const unlearnAPI = {
  unlearn: (data) => api.post('/api/unlearn/', data),
};

// Context API
export const contextAPI = {
  getAll: (agentId = null) => {
    const params = agentId ? { agent_id: agentId } : {};
    return api.get('/api/contexts/', { params });
  },
  create: (data) => api.post('/api/contexts/', data),
};

export default api;
