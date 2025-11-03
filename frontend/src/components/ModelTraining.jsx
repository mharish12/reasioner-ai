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
    modelType: 'rag',
    modelName: '',
    files: [],
    plainText: '',
  });
  const [filePreviews, setFilePreviews] = useState([]);

  const modelTypes = [
    { value: 'xgboost', label: 'XGBoost (Classification/Regression)' },
    { value: 'rag', label: 'RAG (Retrieval Augmented Generation)' },
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

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('agent_id', selectedAgent.id);
      formData.append('model_type', trainingData.modelType);
      formData.append('model_name', trainingData.modelName);

      if (trainingData.files.length > 0) {
        trainingData.files.forEach((file) => {
          formData.append('files', file);
        });
      }

      if (trainingData.plainText.trim()) {
        formData.append('plain_text', trainingData.plainText);
      }

      await modelsAPI.train(formData);

      // Reset form
      setTrainingData({
        modelType: 'rag',
        modelName: '',
        files: [],
        plainText: '',
      });
      setFilePreviews([]);

      onModelTrained();
      alert('Model training started successfully!');
    } catch (error) {
      console.error('Error training model:', error);
      alert(
        'Failed to train model: ' +
          (error.response?.data?.detail || error.message)
      );
    } finally {
      setLoading(false);
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
                className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                Start Training
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

export default ModelTraining;
