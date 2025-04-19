import Papa from 'papaparse';
import { logEnvironmentInfo, validateDataset, measurePerformance } from './debugHelper';

export interface DataPoint {
  [key: string]: string | number;
}

export const loadDataset = async (datasetName: string): Promise<DataPoint[]> => {
  logEnvironmentInfo();
  
  // Use the measuredPerformance utility to time the data loading
  return await measurePerformance(`Loading ${datasetName}`, async () => {
    let path = '';
    
    switch (datasetName) {
      case 'Adult Income':
        path = 'adult_.csv';
        break;
      case 'Heart Disease':
        path = 'heart.csv';
        break;
      case 'Loan Approval':
        path = 'loan.csv';
        break;
      default:
        throw new Error('Invalid dataset');
    }

    // Try multiple possible paths for different deployment environments
    const possiblePaths = [
      `/src/data/${path}`,          // Local development with Vite
      `/data/${path}`,              // Some deployments
      `/${path}`,                   // Root directory
      `https://raw.githubusercontent.com/farazjawedd/XAI_FINAL_PROJ/main/public/data/${path}` // Direct GitHub link as fallback
    ];
    
    let csvText = '';
    let loadError: Error | null = null;
    
    // Try each possible path until one works
    for (const testPath of possiblePaths) {
      try {
        console.log(`Attempting to load from: ${testPath}`);
        const response = await fetch(testPath, { cache: 'no-store' });
        if (!response.ok) {
          console.log(`Failed to load from ${testPath}: ${response.status}`);
          continue;
        }
        
        csvText = await response.text();
        if (csvText && csvText.length > 0) {
          console.log(`Successfully loaded ${csvText.length} bytes from ${testPath}`);
          break;
        }
      } catch (e) {
        console.log(`Error loading from ${testPath}:`, e);
        loadError = e instanceof Error ? e : new Error(String(e));
      }
    }
    
    if (!csvText) {
      throw loadError || new Error(`Failed to load dataset ${datasetName} from any path`);
    }
    
    const data = await new Promise<DataPoint[]>((resolve, reject) => {
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          resolve(results.data as DataPoint[]);
        },
        error: (error) => {
          reject(error);
        }
      });
    });

    // Before returning, validate the dataset
    if (!validateDataset(data)) {
      throw new Error(`Invalid dataset: ${datasetName}`);
    }
    
    return data;
  });
};

const formatFeatureName = (feature: string): string => {
  const featureMap: { [key: string]: string } = {
    // Heart Disease features
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression',
    'ca': 'Number of Vessels',
    'thal': 'Thalassemia',
    
    // Loan features
    'applicant_income': 'Monthly Income',
    'loan_amount': 'Loan Amount',
    'loan_term': 'Loan Term',
    'credit_history': 'Credit History',
    
    // Adult Income features
    'capital-gain': 'Capital Gains',
    'capital-loss': 'Capital Losses',
    'hours-per-week': 'Hours per Week',
    'education-num': 'Years of Education'
  };

  return featureMap[feature] || feature.split('_').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
};

export const predictFromTree = (
  tree: any,
  input: { [key: string]: string | number }
): { prediction: string; confidence: number } => {
  let currentNode = tree;

  while (currentNode.children) {
    const feature = currentNode.feature;
    const value = input[feature];
    const threshold = currentNode.threshold;
    const isNumeric = typeof threshold === 'number';

    if (isNumeric) {
      currentNode = Number(value) <= threshold 
        ? currentNode.children[0] 
        : currentNode.children[1];
    } else {
      currentNode = String(value) === String(threshold)
        ? currentNode.children[0]
        : currentNode.children[1];
    }
  }

  return {
    prediction: currentNode.name,
    confidence: currentNode.confidence || 0
  };
};

export const buildDecisionTree = (data: DataPoint[], target: string, maxDepth = 4) => {
  if (maxDepth === 0 || data.length < 5) return null;

  const getEntropy = (subset: DataPoint[]) => {
    const counts: { [key: string]: number } = {};
    subset.forEach(row => {
      const value = String(row[target]);
      counts[value] = (counts[value] || 0) + 1;
    });
    
    return -Object.values(counts).reduce((sum, count) => {
      const p = count / subset.length;
      return sum + p * Math.log2(p);
    }, 0);
  };

  const getDistribution = (subset: DataPoint[]) => {
    const distribution: { [key: string]: number } = {};
    subset.forEach(row => {
      const value = String(row[target]);
      distribution[value] = (distribution[value] || 0) + 1;
    });
    return distribution;
  };

  const findBestSplit = (subset: DataPoint[]) => {
    let bestGain = -Infinity;
    let bestFeature = '';
    let bestThreshold = 0;
    let bestIsNumeric = true;

    const features = Object.keys(subset[0])
      .filter(f => f !== target && f !== 'id' && f !== 'loan_id');
    
    for (const feature of features) {
      const values = subset.map(row => row[feature]);
      const isNumeric = values.every(v => typeof v === 'number');

      if (isNumeric) {
        const numericValues = values as number[];
        const min = Math.min(...numericValues);
        const max = Math.max(...numericValues);
        const steps = Math.min(20, (max - min)); // Adaptive step size
        const step = (max - min) / steps;

        for (let threshold = min + step; threshold < max; threshold += step) {
          const left = subset.filter(row => Number(row[feature]) <= threshold);
          const right = subset.filter(row => Number(row[feature]) > threshold);
          
          if (left.length < 5 || right.length < 5) continue;

          const entropyBefore = getEntropy(subset);
          const entropyAfter = (left.length / subset.length) * getEntropy(left) +
                             (right.length / subset.length) * getEntropy(right);
          const gain = entropyBefore - entropyAfter;

          if (gain > bestGain) {
            bestGain = gain;
            bestFeature = feature;
            bestThreshold = threshold;
            bestIsNumeric = true;
          }
        }
      } else {
        const categories = new Set(values.map(String));
        
        for (const category of categories) {
          const left = subset.filter(row => String(row[feature]) === category);
          const right = subset.filter(row => String(row[feature]) !== category);

          if (left.length < 5 || right.length < 5) continue;

          const entropyBefore = getEntropy(subset);
          const entropyAfter = (left.length / subset.length) * getEntropy(left) +
                             (right.length / subset.length) * getEntropy(right);
          const gain = entropyBefore - entropyAfter;

          if (gain > bestGain) {
            bestGain = gain;
            bestFeature = feature;
            bestThreshold = category;
            bestIsNumeric = false;
          }
        }
      }
    }

    return { 
      feature: bestFeature, 
      threshold: bestThreshold, 
      gain: bestGain, 
      isNumeric: bestIsNumeric 
    };
  };

  const getMajorityClass = (subset: DataPoint[]) => {
    const counts: { [key: string]: number } = {};
    subset.forEach(row => {
      const value = String(row[target]);
      counts[value] = (counts[value] || 0) + 1;
    });
    return Object.entries(counts).reduce((a, b) => a[1] > b[1] ? a : b)[0];
  };

  const split = findBestSplit(data);
  const distribution = getDistribution(data);
  
  if (split.gain <= 0.01) {
    const majorityClass = getMajorityClass(data);
    const confidence = data.filter(row => String(row[target]) === majorityClass).length / data.length;
    
    return {
      name: majorityClass,
      value: 1,
      confidence,
      samples: data.length,
      distribution,
      prediction: true
    };
  }

  let left: DataPoint[];
  let right: DataPoint[];
  let condition: string;
  const formattedFeature = formatFeatureName(split.feature);

  if (split.isNumeric) {
    const threshold = Number(split.threshold).toFixed(1);
    left = data.filter(row => Number(row[split.feature]) <= split.threshold);
    right = data.filter(row => Number(row[split.feature]) > split.threshold);
    condition = `≤ ${threshold}`;
  } else {
    left = data.filter(row => String(row[split.feature]) === split.threshold);
    right = data.filter(row => String(row[split.feature]) !== split.threshold);
    condition = `= "${split.threshold}"`;
  }

  const leftChild = buildDecisionTree(left, target, maxDepth - 1);
  const rightChild = buildDecisionTree(right, target, maxDepth - 1);

  if (!leftChild || !rightChild) {
    const majorityClass = getMajorityClass(data);
    const confidence = data.filter(row => String(row[target]) === majorityClass).length / data.length;
    
    return {
      name: majorityClass,
      value: 1,
      confidence,
      samples: data.length,
      distribution,
      prediction: true
    };
  }

  return {
    name: formattedFeature,
    condition,
    confidence: split.gain,
    feature: formattedFeature,
    threshold: split.threshold,
    samples: data.length,
    distribution,
    children: [leftChild, rightChild]
  };
};