/**
 * Helper utilities for debugging data loading and processing issues
 */

export const logEnvironmentInfo = () => {
  // Log environment information to help debug deployment issues
  console.log('Environment:', {
    isProduction: process.env.NODE_ENV === 'production',
    baseUrl: window.location.origin,
    deploymentPlatform: typeof window !== 'undefined' && window.location.hostname.includes('vercel.app') 
      ? 'Vercel' 
      : 'Other/Local'
  });
};

export const validateDataset = (data: any[] | null) => {
  if (!data) {
    console.error('Dataset is null');
    return false;
  }
  
  if (!Array.isArray(data)) {
    console.error('Dataset is not an array, got:', typeof data);
    return false;
  }
  
  if (data.length === 0) {
    console.error('Dataset is empty');
    return false;
  }
  
  // Log the first few rows to help with debugging
  console.log('First 3 rows:', data.slice(0, 3));
  console.log('Total rows:', data.length);
  
  return true;
};

export const measurePerformance = async (name: string, fn: () => Promise<any>) => {
  console.time(name);
  try {
    const result = await fn();
    console.timeEnd(name);
    return result;
  } catch (error) {
    console.timeEnd(name);
    console.error(`Error in ${name}:`, error);
    throw error;
  }
};
