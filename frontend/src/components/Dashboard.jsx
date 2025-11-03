import React, { useState, useEffect } from 'react';
import { agentsAPI, modelsAPI, queryAPI, unlearnAPI } from '../services/api';
import AgentManagement from './AgentManagement';
import ModelTraining from './ModelTraining';
import ModelQuery from './ModelQuery';
import { Brain, Database, Settings, Activity } from 'lucide-react';

function Dashboard() {
  const [activeTab, setActiveTab] = useState('agents');
  const [agents, setAgents] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchAgents = async () => {
    try {
      const response = await agentsAPI.getAll();
      setAgents(response.data);
      if (response.data.length > 0 && !selectedAgent) {
        setSelectedAgent(response.data[0]);
      }
    } catch (error) {
      console.error('Error fetching agents:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await modelsAPI.getAll(selectedAgent?.id);
      setModels(response.data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  useEffect(() => {
    fetchAgents();
  }, []);

  useEffect(() => {
    if (selectedAgent) {
      fetchModels();
    }
  }, [selectedAgent]);

  const tabs = [
    { id: 'agents', name: 'Agents', icon: Settings },
    { id: 'training', name: 'Model Training', icon: Brain },
    { id: 'query', name: 'Query Models', icon: Activity },
    { id: 'database', name: 'Database', icon: Database },
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-64 bg-gray-900 text-white flex flex-col">
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-2xl font-bold">AI Platform</h1>
          <p className="text-gray-400 text-sm mt-1">Model Training</p>
        </div>

        <nav className="flex-1 p-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg mb-2 transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-800'
                }`}
              >
                <Icon size={20} />
                <span>{tab.name}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-gray-800">
          <div className="text-xs text-gray-400">
            <p>Status: {loading ? 'Processing...' : 'Ready'}</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-8">
          {activeTab === 'agents' && (
            <AgentManagement
              agents={agents}
              selectedAgent={selectedAgent}
              onAgentSelect={setSelectedAgent}
              onAgentCreated={fetchAgents}
              onAgentDeleted={fetchAgents}
            />
          )}

          {activeTab === 'training' && (
            <ModelTraining
              agents={agents}
              selectedAgent={selectedAgent}
              onAgentSelect={setSelectedAgent}
              onModelTrained={fetchModels}
              setLoading={setLoading}
            />
          )}

          {activeTab === 'query' && (
            <ModelQuery
              agents={agents}
              models={models}
              selectedAgent={selectedAgent}
              onAgentSelect={setSelectedAgent}
            />
          )}

          {activeTab === 'database' && (
            <DatabaseView
              agents={agents}
              models={models}
              selectedAgent={selectedAgent}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function DatabaseView({ agents, models, selectedAgent }) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4">Database Overview</h2>

        <div className="grid grid-cols-2 gap-6 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="text-3xl font-bold text-blue-600">
              {agents.length}
            </div>
            <div className="text-gray-600">Total Agents</div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <div className="text-3xl font-bold text-green-600">
              {models.length}
            </div>
            <div className="text-gray-600">Trained Models</div>
          </div>
        </div>

        {selectedAgent && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-3">
              Current Agent: {selectedAgent.name}
            </h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <pre className="text-sm overflow-auto">
                {JSON.stringify(selectedAgent, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
