import React, { useState, useEffect } from 'react';
import { modelsAPI, queryAPI } from '../services/api';
import { Send, Loader } from 'lucide-react';

function ModelQuery({ agents, models, selectedAgent, onAgentSelect }) {
  const [selectedModel, setSelectedModel] = useState(null);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [queryHistory, setQueryHistory] = useState([]);

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      // Filter only completed models
      const completedModels = models.filter((m) => m.status === 'completed');
      if (completedModels.length > 0) {
        setSelectedModel(completedModels[0]);
      }
    }
  }, [models]);

  const fetchModels = async () => {
    if (selectedAgent) {
      try {
        const response = await modelsAPI.getAll(selectedAgent.id);
        const completedModels = response.data.filter(
          (m) => m.status === 'completed'
        );
        if (completedModels.length > 0 && !selectedModel) {
          setSelectedModel(completedModels[0]);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    }
  };

  useEffect(() => {
    fetchModels();
  }, [selectedAgent]);

  const handleQuery = async (e) => {
    e.preventDefault();

    if (!selectedModel) {
      alert('Please select a trained model');
      return;
    }

    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    setLoading(true);
    try {
      const result = await queryAPI.query({
        model_id: selectedModel.id,
        query_text: query,
      });
      setResponse(result.data.response);

      // Add to history
      setQueryHistory((prev) => [
        {
          query,
          response: result.data.response,
          timestamp: result.data.timestamp,
        },
        ...prev,
      ]);

      setQuery('');
    } catch (error) {
      console.error('Error querying model:', error);
      alert(
        'Failed to query model: ' +
          (error.response?.data?.detail || error.message)
      );
    } finally {
      setLoading(false);
    }
  };

  const completedModels = models.filter((m) => m.status === 'completed');

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">Query Models</h2>
        <p className="text-gray-600">Ask questions to your trained models</p>
      </div>

      {!selectedAgent ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <p className="text-yellow-800">
            Please select an agent first from the Agents tab
          </p>
        </div>
      ) : completedModels.length === 0 ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <p className="text-yellow-800">
            No trained models available for agent "{selectedAgent.name}". Please
            train a model first.
          </p>
        </div>
      ) : (
        <>
          {/* Model Selection */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="mb-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                <p className="text-blue-800">
                  <strong>Selected Agent:</strong> {selectedAgent.name}
                </p>
              </div>
            </div>

            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model *
            </label>
            <select
              value={selectedModel?.id || ''}
              onChange={(e) => {
                const model = completedModels.find(
                  (m) => m.id === parseInt(e.target.value)
                );
                setSelectedModel(model);
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">-- Select a model --</option>
              {completedModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.model_name} ({model.model_type})
                </option>
              ))}
            </select>

            {selectedModel && (
              <div className="mt-4 bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Model Type:</span>{' '}
                    {selectedModel.model_type}
                  </div>
                  <div>
                    <span className="font-medium">Documents:</span>{' '}
                    {selectedModel.training_documents_count}
                  </div>
                  {selectedModel.accuracy && (
                    <div>
                      <span className="font-medium">Accuracy:</span>{' '}
                      {(selectedModel.accuracy * 100).toFixed(2)}%
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Query Interface */}
          {selectedModel && (
            <div className="bg-white rounded-lg shadow p-6">
              <form onSubmit={handleQuery} className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Enter Your Query
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows="4"
                  placeholder="Ask your model a question..."
                />
                <button
                  type="submit"
                  disabled={loading}
                  className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <Loader className="animate-spin" size={20} />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <Send size={20} />
                      <span>Send Query</span>
                    </>
                  )}
                </button>
              </form>

              {/* Response Display */}
              {response && (
                <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h3 className="font-semibold mb-2 text-blue-900">
                    Response:
                  </h3>
                  <p className="text-gray-800 whitespace-pre-wrap">
                    {response}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Query History */}
          {queryHistory.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-xl font-semibold mb-4">Query History</h3>
              <div className="space-y-4">
                {queryHistory.map((item, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-4"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <p className="font-medium text-gray-900">{item.query}</p>
                      <span className="text-xs text-gray-500">
                        {new Date(item.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-gray-700 text-sm bg-gray-50 p-3 rounded whitespace-pre-wrap">
                      {item.response}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default ModelQuery;
