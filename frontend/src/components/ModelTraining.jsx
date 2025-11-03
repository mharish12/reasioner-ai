import React, { useState } from 'react';
import { modelsAPI } from '../services/api';
import { Upload, FileText, X } from 'lucide-react';

function ModelTraining({
  agents,
  selectedAgent,
  onAgentSelect,
  onModelTrained,
  setLoading,
}) {
  const [trainingData, setTrainingData] = useState({
    modelType: 'langchain_rag', // Default to new LangChain RAG
    modelName: '',
    files: [],
    plainText: '',
    llmType: 'ollama', // LLM type for langchain_rag
    llmModel: 'llama2', // LLM model name
    llmBaseUrl: 'http://localhost:11434', // Ollama base URL
  });
  const [filePreviews, setFilePreviews] = useState([]);
  const [isTraining, setIsTraining] = useState(false); // Local loading state for button

  const modelTypes = [
    { value: 'xgboost', label: 'XGBoost (Classification/Regression)' },
    {
      value: 'rag',
      label: 'RAG - FAISS (Retrieval Augmented Generation - Legacy)',
    },
    {
      value: 'langchain_rag',
      label: 'RAG - LangChain (pgvector + LLM - Recommended)',
    },
    { value: 'transformer', label: 'Transformer (Text Generation)' },
  ];

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    setTrainingData({
      ...trainingData,
      files: [...trainingData.files, ...files],
    });

    const newPreviews = files.map((file) => ({
      name: file.name,
      size: file.size,
      type: file.type,
    }));
    setFilePreviews([...filePreviews, ...newPreviews]);
  };

  const handleRemoveFile = (index) => {
    const newFiles = trainingData.files.filter((_, i) => i !== index);
    const newPreviews = filePreviews.filter((_, i) => i !== index);
    setTrainingData({ ...trainingData, files: newFiles });
    setFilePreviews(newPreviews);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedAgent) {
      alert('Please select an agent first');
      return;
    }

    if (trainingData.files.length === 0 && !trainingData.plainText.trim()) {
      alert('Please provide either files or plain text for training');
      return;
    }

    if (!trainingData.modelName.trim()) {
      alert('Please provide a model name');
      return;
    }

    console.log('='.repeat(80));
    console.log('Training Request Started');
    console.log('Agent:', selectedAgent.name, '(ID:', selectedAgent.id + ')');
    console.log('Model Type:', trainingData.modelType);
    console.log('Model Name:', trainingData.modelName);
    console.log('Files:', trainingData.files.length);
    console.log('Plain Text Length:', trainingData.plainText.length, 'chars');

    setLoading(true);
    setIsTraining(true); // Disable button

    try {
      const formData = new FormData();
      formData.append('agent_id', selectedAgent.id);
      formData.append('model_type', trainingData.modelType);
      formData.append('model_name', trainingData.modelName);

      // Add parameters for langchain_rag
      if (trainingData.modelType === 'langchain_rag') {
        const parameters = {
          llm_type: trainingData.llmType,
          llm_config: {
            model_name: trainingData.llmModel,
            base_url: trainingData.llmBaseUrl,
            temperature: 0.7,
          },
          embedding_model: 'all-MiniLM-L6-v2',
          top_k: 3,
        };
        console.log('LangChain RAG Parameters:', parameters);
        formData.append('parameters', JSON.stringify(parameters));
      }

      if (trainingData.files.length > 0) {
        console.log(
          'Uploading files:',
          trainingData.files.map((f) => f.name).join(', ')
        );
        trainingData.files.forEach((file) => {
          formData.append('files', file);
        });
      }

      if (trainingData.plainText.trim()) {
        console.log('Including plain text input');
        formData.append('plain_text', trainingData.plainText);
      }

      console.log('Sending training request to backend...');
      const startTime = Date.now();
      const response = await modelsAPI.train(formData);
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`Training request completed in ${duration} seconds`);
      console.log('Model ID:', response.id);
      console.log('Model Status:', response.status);

      // Reset form
      setTrainingData({
        modelType: 'langchain_rag', // Default to new LangChain RAG
        modelName: '',
        files: [],
        plainText: '',
        llmType: 'ollama',
        llmModel: 'llama2',
        llmBaseUrl: 'http://localhost:11434',
      });
      setFilePreviews([]);

      onModelTrained();
      console.log('Training completed successfully!');
      console.log('='.repeat(80));
      alert('Model training started successfully!');
    } catch (error) {
      console.error('='.repeat(80));
      console.error('Training Error:', error);
      console.error('Error Details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
      });
      console.error('='.repeat(80));
      alert(
        'Failed to train model: ' +
          (error.response?.data?.detail || error.message)
      );
    } finally {
      setLoading(false);
      setIsTraining(false); // Re-enable button
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">Model Training</h2>
        <p className="text-gray-600">
          Train AI models using documents or plain text
        </p>
      </div>

      {!selectedAgent ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <p className="text-yellow-800">
            Please select or create an agent first from the Agents tab
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow p-8">
          <div className="mb-6">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <p className="text-blue-800">
                <strong>Selected Agent:</strong> {selectedAgent.name}
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Model Type Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model Type *
              </label>
              <select
                value={trainingData.modelType}
                onChange={(e) =>
                  setTrainingData({
                    ...trainingData,
                    modelType: e.target.value,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {modelTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Model Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Name *
              </label>
              <input
                type="text"
                required
                value={trainingData.modelName}
                onChange={(e) =>
                  setTrainingData({
                    ...trainingData,
                    modelName: e.target.value,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="e.g., My RAG Model v1"
              />
            </div>

            {/* LangChain RAG Parameters */}
            {trainingData.modelType === 'langchain_rag' && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-4">
                <h3 className="text-sm font-semibold text-blue-900 mb-3">
                  LangChain RAG Configuration
                </h3>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    LLM Type *
                  </label>
                  <select
                    value={trainingData.llmType}
                    onChange={(e) =>
                      setTrainingData({
                        ...trainingData,
                        llmType: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="ollama">Ollama (Free, Local)</option>
                    <option value="openai">OpenAI (Paid, Cloud)</option>
                    <option value="huggingface">
                      HuggingFace (Free, Local)
                    </option>
                  </select>
                </div>

                {trainingData.llmType === 'ollama' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Ollama Model Name *
                      </label>
                      <input
                        type="text"
                        value={trainingData.llmModel}
                        onChange={(e) =>
                          setTrainingData({
                            ...trainingData,
                            llmModel: e.target.value,
                          })
                        }
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="e.g., llama2, mistral, codellama"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Make sure the model is downloaded: ollama pull{' '}
                        {trainingData.llmModel}
                      </p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Ollama Base URL
                      </label>
                      <input
                        type="text"
                        value={trainingData.llmBaseUrl}
                        onChange={(e) =>
                          setTrainingData({
                            ...trainingData,
                            llmBaseUrl: e.target.value,
                          })
                        }
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="http://localhost:11434"
                      />
                    </div>
                  </>
                )}

                {trainingData.llmType === 'openai' && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                    <p className="text-sm text-yellow-800">
                      <strong>Note:</strong> OpenAI requires an API key. Set the{' '}
                      <code className="bg-yellow-100 px-1 rounded">
                        OPENAI_API_KEY
                      </code>{' '}
                      environment variable on the backend server.
                    </p>
                  </div>
                )}

                {trainingData.llmType === 'huggingface' && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                    <p className="text-sm text-yellow-800">
                      <strong>Note:</strong> HuggingFace models are free but
                      slower. They require more memory and may take longer to
                      load.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Documents
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                <input
                  type="file"
                  multiple
                  accept=".xlsx,.xls,.csv,.txt,.pdf,.jpg,.jpeg,.png,.gif,.bmp,.tiff,.tif,.webp"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="mx-auto mb-2 text-gray-400" size={48} />
                  <p className="text-gray-600 mb-1">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-gray-400">
                    Excel, CSV, TXT, PDF, Images supported
                  </p>
                </label>
              </div>

              {filePreviews.length > 0 && (
                <div className="mt-4 space-y-2">
                  {filePreviews.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-gray-50 rounded-lg p-3"
                    >
                      <div className="flex items-center space-x-3">
                        <FileText className="text-blue-600" size={20} />
                        <div>
                          <p className="font-medium">{file.name}</p>
                          <p className="text-sm text-gray-500">
                            {(file.size / 1024).toFixed(2)} KB
                          </p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRemoveFile(index)}
                        className="text-red-600 hover:text-red-800"
                      >
                        <X size={20} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Plain Text Input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Or Enter Plain Text
              </label>
              <textarea
                value={trainingData.plainText}
                onChange={(e) =>
                  setTrainingData({
                    ...trainingData,
                    plainText: e.target.value,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows="8"
                placeholder="Enter your training text here..."
              />
            </div>

            {/* Submit Button */}
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={isTraining || !selectedAgent}
                className={`px-8 py-3 rounded-lg transition-colors font-medium ${
                  isTraining || !selectedAgent
                    ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {isTraining ? 'Training...' : 'Start Training'}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

export default ModelTraining;
