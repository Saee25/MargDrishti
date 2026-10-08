import { useState, useEffect, useCallback } from 'react';

const cache = new Map();

export function useApi(apiFunc, ...args) {
  const cacheKey = apiFunc.name + JSON.stringify(args);
  
  const [data, setData] = useState(cache.get(cacheKey) || null);
  const [loading, setLoading] = useState(!cache.has(cacheKey));
  const [error, setError] = useState(null);

  const fetch = useCallback(async (ignoreCache = false) => {
    if (!ignoreCache && cache.has(cacheKey)) {
      setData(cache.get(cacheKey));
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await apiFunc(...args);
      cache.set(cacheKey, result);
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [cacheKey, apiFunc, args]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: () => fetch(true) };
}
