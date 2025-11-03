import React, { useState } from 'react';
import { agentsAPI } from '../services/api';
import { Plus, Trash2, Check } from 'lucide-react';

function AgentManagement({
  agents,
  selectedAgent,
  onAgentSelect,
  onAgentCreated,
  onAgentDeleted,
}) {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newAgent, setNewAgent] = useState({ name: '', description: '' });
  const [loading, setLoading] = useState(false);

  const handleCreateAgent = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await agentsAPI.create(newAgent);
      setNewAgent({ name: '', description: '' });
      setShowCreateForm(false);
      onAgentCreated();
    } catch (error) {
      console.error('Error creating agent:', error);
      alert('Failed to create agent');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAgent = async (id) => {
    if (
      !confirm(
        'Are you sure you want to delete this agent? This will also delete all associated models.'
      )
    ) {
      return;
    }

    try {
      await agentsAPI.delete(id);
      onAgentDeleted();
      if (selectedAgent?.id === id) {
        onAgentSelect(null);
      }
    } catch (error) {
      console.error('Error deleting agent:', error);
      alert('Failed to delete agent');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold">Agents Management</h2>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Plus size={20} />
          <span>Create Agent</span>
        </button>
      </div>

      {showCreateForm && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold mb-4">Create New Agent</h3>
          <form onSubmit={handleCreateAgent} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Agent Name *
              </label>
              <input
                type="text"
                required
                value={newAgent.name}
                onChange={(e) =>
                  setNewAgent({ ...newAgent, name: e.target.value })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter agent name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={newAgent.description}
                onChange={(e) =>
                  setNewAgent({ ...newAgent, description: e.target.value })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows="3"
                placeholder="Enter agent description"
              />
            </div>
            <div className="flex space-x-3">
              <button
                type="submit"
                disabled={loading}
                className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <Check size={20} />
                <span>{loading ? 'Creating...' : 'Create Agent'}</span>
              </button>
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className={`bg-white rounded-lg shadow p-6 cursor-pointer transition-all ${
              selectedAgent?.id === agent.id
                ? 'ring-2 ring-blue-600'
                : 'hover:shadow-lg'
            }`}
            onClick={() => onAgentSelect(agent)}
          >
            <div className="flex justify-between items-start mb-3">
              <h3 className="text-xl font-semibold">{agent.name}</h3>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteAgent(agent.id);
                }}
                className="text-red-600 hover:text-red-800 transition-colors"
              >
                <Trash2 size={18} />
              </button>
            </div>
            {agent.description && (
              <p className="text-gray-600 text-sm mb-4">{agent.description}</p>
            )}
            <div className="text-xs text-gray-400">
              Created: {new Date(agent.created_at).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>

      {agents.length === 0 && (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <p className="text-gray-500 text-lg">
            No agents created yet. Create your first agent to get started!
          </p>
        </div>
      )}
    </div>
  );
}

export default AgentManagement;
