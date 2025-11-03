export const downloadModel = async (modelId) => {
  // Placeholder for model download functionality
  console.log('Download model:', modelId);
};

export const deleteModel = async (modelId) => {
  // Placeholder for model deletion
  console.log('Delete model:', modelId);
};

export const exportQueryHistory = (history) => {
  const blob = new Blob([JSON.stringify(history, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'query_history.json';
  a.click();
  URL.revokeObjectURL(url);
};
