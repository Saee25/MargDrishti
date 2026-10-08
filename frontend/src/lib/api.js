const TIMEOUT = 10000;

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

/** Get list of available models */
export const getModels = () => fetchWithTimeout('/api/models');

/** Get model details */
export const getModel = (modelId) => fetchWithTimeout(`/api/models/${modelId}`);

/** Get summary metrics for all models */
export const getMetricsSummary = () => fetchWithTimeout('/api/metrics/summary');

/** Get detailed metrics for a model */
export const getMetrics = (modelId) => fetchWithTimeout(`/api/metrics/${modelId}`);

/** Get dataset stats */
export const getDatasetStats = () => fetchWithTimeout('/api/dataset/stats');

/** Get random test samples */
export const getSamples = (limit = 12) => fetchWithTimeout(`/api/samples?limit=${limit}`);

/** Predict a sample */
export const predict = (modelId, imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('model', modelId === 'margnet-v5' ? 'margnet' : (modelId === 'resnet50-finetuned' ? 'resnet50' : 'both'));
  return fetchWithTimeout(`/api/predict`, {
    method: 'POST',
    body: formData
  });
};

export const detectYolo = (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  return fetchWithTimeout(`/api/detect`, {
    method: 'POST',
    body: formData
  });
};
