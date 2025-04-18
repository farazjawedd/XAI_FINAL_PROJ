import React, { useState } from 'react';
import { Trees as Tree, Users, Brain, HeartPulse, Wallet } from 'lucide-react';
import DecisionTree from './components/DecisionTree';

interface Dataset {
  name: string;
  description: string;
  icon: React.ReactNode;
  details: string[];
}

function App() {
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);

  const datasets: Dataset[] = [
    {
      name: "Adult Income",
      description: "Predict income levels based on demographic and employment factors",
      icon: <Users className="w-8 h-8" />,
      details: [
        "Predicts if income exceeds $50K/year",
        "Features include age, education, occupation",
        "Based on 1994 Census database",
        "Considers factors like work hours and capital gains"
      ]
    },
    {
      name: "Heart Disease",
      description: "Analyze medical factors to predict heart disease risk",
      icon: <HeartPulse className="w-8 h-8" />,
      details: [
        "Predicts presence of heart disease",
        "Uses 13 clinical features",
        "Includes factors like chest pain type and blood pressure",
        "Based on real medical examination data"
      ]
    },
    {
      name: "Loan Approval",
      description: "Evaluate loan applications based on financial criteria",
      icon: <Wallet className="w-8 h-8" />,
      details: [
        "Predicts loan approval status",
        "Considers income, credit history, and loan amount",
        "Includes both personal and financial factors",
        "Based on historical banking data"
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Tree className="w-8 h-8 text-indigo-600" />
            <h1 className="text-2xl font-bold text-gray-900">Decision Tree Explorer</h1>
          </div>
          <div className="flex items-center space-x-2">
            <Brain className="w-6 h-6 text-gray-500" />
            <span className="text-sm text-gray-500">Real-World Decision Analysis</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Explore Real-World Decision Making
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Analyze how machine learning models make decisions using real datasets. 
            Understand the factors that influence predictions and explore different scenarios.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {datasets.map((dataset) => (
            <button
              key={dataset.name}
              onClick={() => setSelectedDataset(dataset.name)}
              className={`p-6 bg-white rounded-xl shadow-md transition-all hover:shadow-lg border-2 
                ${selectedDataset === dataset.name 
                  ? 'border-indigo-500 ring-2 ring-indigo-500 ring-opacity-50' 
                  : 'border-transparent'}`}
            >
              <div className="flex items-center justify-center mb-4">
                <div className="p-3 bg-indigo-100 rounded-full">
                  {dataset.icon}
                </div>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">{dataset.name}</h3>
              <p className="text-gray-600 mb-4">{dataset.description}</p>
              <ul className="text-sm text-left text-gray-500 space-y-1">
                {dataset.details.map((detail, index) => (
                  <li key={index} className="flex items-start">
                    <span className="mr-2">•</span>
                    <span>{detail}</span>
                  </li>
                ))}
              </ul>
            </button>
          ))}
        </div>

        {selectedDataset && (
          <div className="mt-12">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              <div className="p-6 border-b border-gray-200">
                <h3 className="text-2xl font-bold text-gray-900 mb-2">
                  {selectedDataset} Analysis
                </h3>
                <p className="text-gray-600">
                  Explore how different factors influence the prediction for {selectedDataset.toLowerCase()}. 
                  Adjust the values using the interactive controls and see how the decision tree responds.
                </p>
              </div>
              <div className="p-6">
                <DecisionTree dataset={selectedDataset} />
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="bg-white mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-gray-500">
            Using real-world datasets to understand decision-making processes
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;