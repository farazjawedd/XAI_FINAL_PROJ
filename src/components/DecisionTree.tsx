import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { loadDataset, buildDecisionTree, DataPoint, predictFromTree } from '../utils/dataLoader';
import { Sliders, Download } from 'lucide-react';

interface TreeNode {
  name: string;
  children?: TreeNode[];
  value?: number;
  condition?: string;
  confidence?: number;
  feature?: string;
  threshold?: number;
  samples?: number;
  distribution?: { [key: string]: number };
  isNumeric?: boolean;
}

interface DecisionTreeProps {
  dataset: string;
}

interface FeatureInput {
  name: string;
  value: string | number;
  type: 'numeric' | 'categorical';
  options?: string[];
  min?: number;
  max?: number;
}

const DecisionTree: React.FC<DecisionTreeProps> = ({ dataset }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const featureImportanceRef = useRef<SVGSVGElement>(null); // Ref for feature importance chart
  const [selectedPath, setSelectedPath] = useState<d3.HierarchyNode<TreeNode>[]>([]);
  const [hoveredNode, setHoveredNode] = useState<d3.HierarchyNode<TreeNode> | null>(null);
  const [treeData, setTreeData] = useState<TreeNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 800 });
  const [features, setFeatures] = useState<FeatureInput[]>([]);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<{ rows: number; columns: number } | null>(null);
  const [featureImportance, setFeatureImportance] = useState<{ feature: string; importance: number }[]>([]);
  const [inputData, setInputData] = useState<{ [key: string]: string | number }>({});

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        const data = await loadDataset(dataset);
        
        // Ensure we have data before proceeding
        if (!data || data.length === 0) {
          setError("No data available or empty dataset");
          setLoading(false);
          return;
        }

        // Correctly identify the target column for this dataset
        const targetColumn = getTargetColumn(dataset);
        const tree = buildDecisionTree(data, targetColumn);
        setTreeData(tree);

        // Update dataset summary to reflect the actual rows and columns
        setDatasetSummary({
          rows: data.length,
          columns: Object.keys(data[0]).length,
        });

        const featureInputs = extractFeatureInputs(data);
        setFeatures(featureInputs);

        // Initialize input data with default feature values
        const defaultInputData = Object.fromEntries(
          featureInputs.map((f) => [f.name, f.value])
        );
        setInputData(defaultInputData);

        const importance = calculateFeatureImportance(tree);
        setFeatureImportance(importance);
      } catch (err) {
        console.error("Error loading dataset:", err);
        setError(err instanceof Error ? err.message : 'Failed to load dataset');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [dataset]);

  const getTargetColumn = (dataset: string): string => {
    switch (dataset) {
      case 'Adult Income':
        return 'income';
      case 'Heart Disease':
        return 'target';
      case 'Loan Approval':
        return 'loan_status';
      default:
        return 'income';
    }
  };

  const extractFeatureInputs = (data: DataPoint[]): FeatureInput[] => {
    const features: FeatureInput[] = [];
    const firstRow = data[0];
    
    // Dataset-specific excluded columns
    let excludeColumns: string[] = [];
    
    switch(dataset) {
      case 'Adult Income':
        excludeColumns = ['fnlwgt', 'income', 'educational-num'];
        break;
      case 'Heart Disease':
        excludeColumns = ['target'];
        break;
      case 'Loan Approval':
        excludeColumns = ['loan_id', 'loan_status'];
        break;
      default:
        excludeColumns = ['fnlwgt', 'income', 'educational-num'];
    }

    for (const [key, value] of Object.entries(firstRow)) {
      if (excludeColumns.includes(key)) continue;

      // Check if the feature is numeric by looking at all values
      const allValues = data.map(row => row[key]);
      const isNumeric = allValues.every(val => !isNaN(Number(val)) && val !== null && val !== '');

      if (isNumeric) {
        const values = data.map(row => Number(row[key]));
        features.push({
          name: key,
          value: Math.round(d3.mean(values) || 0),
          type: 'numeric',
          min: Math.min(...values),
          max: Math.max(...values)
        });
      } else {
        const uniqueValues = Array.from(new Set(data.map(row => String(row[key]))));
        features.push({
          name: key,
          value: uniqueValues[0],
          type: 'categorical',
          options: uniqueValues
        });
      }
    }

    return features;
  };

  const calculateFeatureImportance = (tree: TreeNode | null): { feature: string; importance: number }[] => {
    if (!tree) return [];
    
    // Track features and their importance
    const importanceMap: { [key: string]: number } = {};
    let totalImportance = 0;

    // Traverse the tree to calculate importance
    const traverse = (node: TreeNode, depth = 0) => {
      if (node.feature) {
        // Weight by samples and depth - nodes higher up are more important
        const nodeImportance = ((node.samples || 1) / 100) * Math.pow(0.9, depth) * (node.confidence || 0.5);
        importanceMap[node.feature] = (importanceMap[node.feature] || 0) + nodeImportance;
        totalImportance += nodeImportance;
      }
      
      if (node.children) {
        node.children.forEach(child => traverse(child, depth + 1));
      }
    };

    traverse(tree);

    // Normalize importance values
    if (totalImportance > 0) {
      Object.keys(importanceMap).forEach(key => {
        importanceMap[key] = importanceMap[key] / totalImportance;
      });
    }

    // Sort by importance
    return Object.entries(importanceMap)
      .map(([feature, importance]) => ({ feature, importance }))
      .sort((a, b) => b.importance - a.importance);
  };

  const handleFeatureChange = (name: string, value: string | number) => {
    // Update the input data state
    const updatedInputData = { ...inputData, [name]: value };
    setInputData(updatedInputData);

    if (treeData) {
      try {
        // Predict the outcome based on the updated input data
        const result = predictFromTree(treeData, updatedInputData);
        setPrediction(result.prediction);
        setConfidence(result.confidence);

        // Highlight the decision path in the tree
        highlightDecisionPath(treeData, updatedInputData);
      } catch (error) {
        console.error("Prediction error:", error);
      }
    }
  };

  const highlightDecisionPath = (tree: TreeNode, input: { [key: string]: string | number }) => {
    const path: TreeNode[] = [];
    let currentNode = tree;

    // Traverse the tree to find the decision path
    while (currentNode.children) {
      path.push(currentNode);
      const feature = currentNode.feature!;
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

    path.push(currentNode); // Add the leaf node
    setSelectedPath(path.map((node) => d3.hierarchy(node))); // Update the selected path

    // Debugging: Log the decision path
    console.log("Decision Path:", path.map((node) => node.name));
  };

  const exportTree = () => {
    const json = JSON.stringify(treeData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${dataset.replace(/\s+/g, '_')}_decision_tree.json`;
    link.click();
  };

  useEffect(() => {
    if (featureImportance.length > 0 && featureImportanceRef.current) {
      // Clear previous chart
      d3.select(featureImportanceRef.current).selectAll("*").remove();

      const margin = { top: 20, right: 30, bottom: 60, left: 120 };
      const width = 600 - margin.left - margin.right;
      const height = 400 - margin.top - margin.bottom;

      const svg = d3
        .select(featureImportanceRef.current)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      // Limit to top 10 features for clarity if there are many
      const topFeatures = featureImportance.slice(0, 10);

      // Scale for x-axis (importance values)
      const x = d3
        .scaleLinear()
        .domain([0, d3.max(topFeatures, (d) => d.importance) || 0.1])
        .range([0, width]);

      // Scale for y-axis (features)
      const y = d3
        .scaleBand()
        .domain(topFeatures.map((d) => d.feature))
        .range([0, height])
        .padding(0.1);

      // Add x-axis
      svg
        .append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".1~%")))
        .selectAll("text")
        .style("font-size", "12px")
        .attr("transform", "rotate(-45)")
        .style("text-anchor", "end");

      // Add x-axis label
      svg.append("text")
        .attr("transform", `translate(${width/2}, ${height + 40})`)
        .style("text-anchor", "middle")
        .text("Relative Importance");

      // Add y-axis
      svg
        .append("g")
        .call(d3.axisLeft(y))
        .selectAll("text")
        .style("font-size", "12px");

      // Add bars
      svg
        .selectAll(".bar")
        .data(topFeatures)
        .join("rect")
        .attr("class", "bar")
        .attr("x", 0)
        .attr("y", (d) => y(d.feature) || 0)
        .attr("width", 0) // Start at 0 for animation
        .attr("height", y.bandwidth())
        .attr("fill", "#4f46e5")
        .transition()
        .duration(800)
        .attr("width", (d) => x(d.importance));
        
      // Add values at the end of bars
      svg
        .selectAll(".value")
        .data(topFeatures)
        .join("text")
        .attr("class", "value")
        .attr("x", (d) => x(d.importance) + 5)
        .attr("y", (d) => (y(d.feature) || 0) + y.bandwidth() / 2)
        .attr("dy", ".35em")
        .text((d) => (d.importance * 100).toFixed(1) + "%")
        .style("font-size", "10px")
        .style("fill", "#4f46e5");
    }
  }, [featureImportance]);

  useEffect(() => {
    if (!svgRef.current || !treeData || loading) return;

    // Clear and re-render the tree visualization
    d3.select(svgRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 120, bottom: 40, left: 120 };
    const width = dimensions.width - margin.left - margin.right;
    const height = dimensions.height - margin.top - margin.bottom;

    const svg = d3
      .select(svgRef.current)
      .attr("width", dimensions.width)
      .attr("height", dimensions.height)
      .call(
        d3.zoom().scaleExtent([0.5, 2]).on("zoom", (event) => {
          svgGroup.attr("transform", event.transform);
        })
      )
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const svgGroup = svg.append("g");

    const treemap = d3.tree<TreeNode>().size([height, width]).nodeSize([60, 120]);

    const root = d3.hierarchy(treeData, (d) => d.children);
    const treeLayout = treemap(root);

    const diagonal = d3
      .linkHorizontal<any, any>()
      .x((d) => d.y)
      .y((d) => d.x);

    // Add links
    svgGroup
      .append("g")
      .attr("class", "links")
      .selectAll(".link")
      .data(treeLayout.links())
      .join("path")
      .attr("class", "link")
      .attr("d", diagonal)
      .style("fill", "none")
      .style("stroke", (d) =>
        selectedPath.some((node) => node.data === d.target.data) ? "#4f46e5" : "#ccc"
      )
      .style("stroke-width", (d) =>
        selectedPath.some((node) => node.data === d.target.data) ? "3px" : "2px"
      );

    // Add nodes
    const nodes = svgGroup
      .append("g")
      .attr("class", "nodes")
      .selectAll(".node")
      .data(treeLayout.descendants())
      .join("g")
      .attr("class", (d) => "node" + (d.children ? " node--internal" : " node--leaf"))
      .attr("transform", (d) => `translate(${d.y},${d.x})`);

    nodes
      .append("circle")
      .attr("r", (d) => {
        const confidence = d.data.confidence || 0.5;
        return 10 + confidence * 10;
      })
      .style("fill", (d) => {
        const confidence = d.data.confidence || 0.5;
        return selectedPath.some((node) => node.data === d.data)
          ? "#4f46e5"
          : d3.interpolateBlues(confidence);
      })
      .style("stroke", "#4f46e5")
      .style("stroke-width", "2px")
      .style("cursor", "pointer");

    nodes
      .append("text")
      .attr("dy", "-1.2em")
      .attr("x", (d) => (d.children ? -13 : 13))
      .style("text-anchor", (d) => (d.children ? "end" : "start"))
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text((d) => (d.data.feature ? `${d.data.feature}` : d.data.name));

    nodes
      .append("text")
      .attr("dy", "0em")
      .attr("x", (d) => (d.children ? -13 : 13))
      .style("text-anchor", (d) => (d.children ? "end" : "start"))
      .style("font-size", "11px")
      .text((d) => d.data.condition || "");

    nodes
      .append("text")
      .attr("dy", "1.2em")
      .attr("x", (d) => (d.children ? -13 : 13))
      .style("text-anchor", (d) => (d.children ? "end" : "start"))
      .style("font-size", "10px")
      .style("fill", "#6b7280")
      .text((d) => {
        const confidence = d.data.confidence;
        return confidence ? `Confidence: ${(confidence * 100).toFixed(1)}%` : "";
      });

    nodes
      .append("text")
      .attr("dy", "2.4em")
      .attr("x", (d) => (d.children ? -13 : 13))
      .style("text-anchor", (d) => (d.children ? "end" : "start"))
      .style("font-size", "10px")
      .style("fill", "#6b7280")
      .text((d) => (d.data.samples ? `Samples: ${d.data.samples}` : ""));
  }, [treeData, selectedPath, loading, dimensions]);

  useEffect(() => {
    const handleResize = () => {
      const container = svgRef.current?.parentElement;
      if (container) {
        setDimensions({
          width: Math.max(container.clientWidth, 1200),
          height: Math.max(container.clientHeight, 800),
        });
      }
    };

    window.addEventListener("resize", handleResize);
    handleResize();

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 p-4 bg-red-50 rounded-lg">
        Error: {error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {datasetSummary && (
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold">Dataset Summary</h3>
          <div className="flex flex-wrap gap-4">
            <p><span className="font-medium">Dataset:</span> {dataset}</p>
            <p><span className="font-medium">Rows:</span> {datasetSummary.rows}</p>
            <p><span className="font-medium">Columns:</span> {datasetSummary.columns}</p>
            {dataset === 'Adult Income' && (
              <p><span className="font-medium">Prediction Target:</span> Income &gt;50K or &lt;=50K</p>
            )}
            {dataset === 'Heart Disease' && (
              <p><span className="font-medium">Prediction Target:</span> Heart Disease Presence (1) or Absence (0)</p>
            )}
            {dataset === 'Loan Approval' && (
              <p><span className="font-medium">Prediction Target:</span> Loan Approved (Y) or Rejected (N)</p>
            )}
          </div>
        </div>
      )}

      {featureImportance.length > 0 && (
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold">Feature Importance</h3>
          <p className="text-sm text-gray-600 mb-2">Relative influence of each feature on the model's predictions</p>
          <svg ref={featureImportanceRef}></svg>
        </div>
      )}

      <div className="bg-white p-4 rounded-lg shadow-lg">
        <button
          onClick={exportTree}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg shadow hover:bg-indigo-700"
        >
          <Download className="w-5 h-5" />
          Export Tree as JSON
        </button>
      </div>

      <div className="bg-white p-4 rounded-lg shadow-lg">
        <div className="flex items-center gap-2 mb-4">
          <Sliders className="w-5 h-5 text-indigo-600" />
          <h3 className="text-lg font-semibold">Interactive Feature Controls:</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature) => (
            <div key={feature.name} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {feature.name}
              </label>
              {feature.type === 'numeric' ? (
                <div className="space-y-1">
                  <input
                    type="range"
                    min={feature.min}
                    max={feature.max}
                    value={inputData[feature.name]}
                    onChange={(e) => handleFeatureChange(feature.name, Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-gray-500">
                    <span>{feature.min}</span>
                    <span>{inputData[feature.name]}</span>
                    <span>{feature.max}</span>
                  </div>
                </div>
              ) : (
                <select
                  value={inputData[feature.name]}
                  onChange={(e) => handleFeatureChange(feature.name, e.target.value)}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                >
                  {feature.options?.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              )}
            </div>
          ))}
        </div>
        {prediction && (
          <div className="mt-4 p-4 bg-indigo-50 rounded-lg">
            <h4 className="font-semibold text-indigo-900">Current Prediction:</h4>
            <p className="text-indigo-700">
              {prediction} 
              {confidence && ` (${(confidence * 100).toFixed(1)}% confidence)`}
            </p>
          </div>
        )}
      </div>

      <div className="bg-white p-4 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-2">How to Read This Decision Tree:</h3>
        <ul className="list-disc pl-5 space-y-2 text-gray-700">
          <li>Each node represents a decision point or final prediction</li>
          <li>Larger circles indicate higher confidence in the decision</li>
          <li>Blue color intensity shows confidence level (darker = more confident)</li>
          <li>Hover over nodes to see detailed information</li>
          <li>Click nodes to highlight the complete decision path</li>
          <li>Use mouse wheel to zoom in/out, drag to pan around</li>
          <li>Adjust feature values above to see how they affect the prediction</li>
        </ul>
      </div>

      <div className="w-full overflow-x-auto bg-white p-4 rounded-lg shadow-inner" style={{ minHeight: '600px' }}>
        <svg 
          ref={svgRef} 
          className="w-full"
          style={{ minWidth: '1200px', minHeight: '800px' }}
          onClick={() => setSelectedPath([])}
        ></svg>
        <div ref={tooltipRef}></div>
      </div>
      
      {hoveredNode && (
        <div className="bg-indigo-50 p-4 rounded-lg">
          <h4 className="font-semibold text-indigo-900">Node Details</h4>
          <p className="text-indigo-700">
            {hoveredNode.data.feature && `Feature: ${hoveredNode.data.feature}`}
            {hoveredNode.data.condition && ` - ${hoveredNode.data.condition}`}
            {hoveredNode.data.name && ` → ${hoveredNode.data.name}`}
            {hoveredNode.data.confidence && 
              ` (${(hoveredNode.data.confidence * 100).toFixed(1)}% confidence)`}
          </p>
        </div>
      )}

      {selectedPath.length > 0 && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h4 className="font-semibold text-gray-900 mb-2">Decision Path</h4>
          <div className="space-y-2">
            {selectedPath.map((node, index) => (
              <div 
                key={index} 
                className="flex items-center space-x-2"
              >
                {index > 0 && <span className="text-gray-400">→</span>}
                <span className="text-gray-700">
                  {node.data.feature ? (
                    <>
                      <span className="font-medium">{node.data.feature}</span>
                      {node.data.condition && 
                        <span className="text-gray-500"> ({node.data.condition})</span>}
                    </>
                  ) : (
                    <span className="font-medium text-indigo-600">
                      Predict: {node.data.name}
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DecisionTree;