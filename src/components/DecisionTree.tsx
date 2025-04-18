import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { loadDataset, buildDecisionTree, DataPoint, predictFromTree } from '../utils/dataLoader';
import { Sliders } from 'lucide-react';

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
  const [selectedPath, setSelectedPath] = useState<d3.HierarchyNode<TreeNode>[]>([]);
  const [hoveredNode, setHoveredNode] = useState<d3.HierarchyNode<TreeNode> | null>(null);
  const [treeData, setTreeData] = useState<TreeNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 800 });
  const [features, setFeatures] = useState<FeatureInput[]>([]);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const data = await loadDataset(dataset);
        const tree = buildDecisionTree(data, getTargetColumn(dataset));
        setTreeData(tree);

        // Extract feature information from the data
        const featureInputs = extractFeatureInputs(data);
        setFeatures(featureInputs);
      } catch (err) {
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
        return '';
    }
  };

  const extractFeatureInputs = (data: DataPoint[]): FeatureInput[] => {
    const features: FeatureInput[] = [];
    const firstRow = data[0];
    const excludeColumns = ['id', 'loan_id', getTargetColumn(dataset)];

    for (const [key, value] of Object.entries(firstRow)) {
      if (excludeColumns.includes(key)) continue;

      const isNumeric = typeof value === 'number';
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

  const handleFeatureChange = (name: string, value: string | number) => {
    setFeatures(prev => 
      prev.map(f => f.name === name ? { ...f, value } : f)
    );

    if (treeData) {
      const inputData = Object.fromEntries(
        features.map(f => [f.name, f.value])
      );
      const result = predictFromTree(treeData, inputData);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
    }
  };

  useEffect(() => {
    if (!svgRef.current || !treeData || loading) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 120, bottom: 40, left: 120 };
    const width = dimensions.width - margin.left - margin.right;
    const height = dimensions.height - margin.top - margin.bottom;

    // Create the SVG container
    const svg = d3.select(svgRef.current)
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create tooltip
    const tooltip = d3.select(tooltipRef.current)
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background-color", "white")
      .style("border", "1px solid #ddd")
      .style("border-radius", "8px")
      .style("padding", "12px")
      .style("box-shadow", "0 2px 4px rgba(0,0,0,0.1)")
      .style("max-width", "300px")
      .style("font-size", "14px")
      .style("z-index", "10");

    // Declares a tree layout and assigns the size
    const treemap = d3.tree<TreeNode>()
      .size([height, width])
      .nodeSize([60, 120]); // Increase node spacing

    // Assigns parent, children, height, depth
    const root = d3.hierarchy(treeData, d => d.children);
    const treeLayout = treemap(root);

    // Creates a curved (diagonal) path from parent to the child nodes
    const diagonal = d3.linkHorizontal<any, any>()
      .x(d => d.y)
      .y(d => d.x);

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 2])
      .on("zoom", (event) => {
        svg.attr("transform", event.transform);
      });

    d3.select(svgRef.current).call(zoom as any);

    // Adds the links between the nodes
    const links = svg.append("g")
      .attr("class", "links")
      .selectAll(".link")
      .data(treeLayout.links())
      .join("path")
      .attr("class", "link")
      .attr("d", diagonal)
      .style("fill", "none")
      .style("stroke", "#ccc")
      .style("stroke-width", "2px");

    // Create node groups
    const nodes = svg.append("g")
      .attr("class", "nodes")
      .selectAll(".node")
      .data(treeLayout.descendants())
      .join("g")
      .attr("class", d => "node" + (d.children ? " node--internal" : " node--leaf"))
      .attr("transform", d => `translate(${d.y},${d.x})`);

    // Add node circles
    nodes.append("circle")
      .attr("r", d => {
        const confidence = d.data.confidence || 0.5;
        return 10 + confidence * 10;
      })
      .style("fill", d => {
        const confidence = d.data.confidence || 0.5;
        return d3.interpolateBlues(confidence);
      })
      .style("stroke", "#4f46e5")
      .style("stroke-width", "2px")
      .style("cursor", "pointer");

    // Add node labels
    nodes.append("text")
      .attr("dy", "-1.2em")
      .attr("x", d => d.children ? -13 : 13)
      .style("text-anchor", d => d.children ? "end" : "start")
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text(d => {
        if (d.data.feature) {
          return `${d.data.feature}`;
        }
        return d.data.name;
      });

    // Add condition/threshold labels
    nodes.append("text")
      .attr("dy", "0em")
      .attr("x", d => d.children ? -13 : 13)
      .style("text-anchor", d => d.children ? "end" : "start")
      .style("font-size", "11px")
      .text(d => d.data.condition || "");

    // Add confidence labels
    nodes.append("text")
      .attr("dy", "1.2em")
      .attr("x", d => d.children ? -13 : 13)
      .style("text-anchor", d => d.children ? "end" : "start")
      .style("font-size", "10px")
      .style("fill", "#6b7280")
      .text(d => {
        const confidence = d.data.confidence;
        return confidence ? `Confidence: ${(confidence * 100).toFixed(1)}%` : '';
      });

    // Add samples count
    nodes.append("text")
      .attr("dy", "2.4em")
      .attr("x", d => d.children ? -13 : 13)
      .style("text-anchor", d => d.children ? "end" : "start")
      .style("font-size", "10px")
      .style("fill", "#6b7280")
      .text(d => d.data.samples ? `Samples: ${d.data.samples}` : '');

    // Enhanced tooltip content
    const showTooltip = (event: MouseEvent, d: d3.HierarchyNode<TreeNode>) => {
      const tooltipContent = `
        <div class="font-semibold mb-2">${d.data.feature ? 'Decision Node' : 'Leaf Node'}</div>
        ${d.data.feature ? `<div class="mb-1">Feature: ${d.data.feature}</div>` : ''}
        ${d.data.condition ? `<div class="mb-1">Condition: ${d.data.condition}</div>` : ''}
        ${d.data.name ? `<div class="mb-1">Prediction: ${d.data.name}</div>` : ''}
        ${d.data.confidence ? `<div class="mb-1">Confidence: ${(d.data.confidence * 100).toFixed(1)}%</div>` : ''}
        ${d.data.samples ? `<div class="mb-1">Samples: ${d.data.samples}</div>` : ''}
        ${d.data.distribution ? `
          <div class="mt-2">
            <div class="font-semibold mb-1">Class Distribution:</div>
            ${Object.entries(d.data.distribution)
              .map(([key, value]) => `<div>${key}: ${value}</div>`)
              .join('')}
          </div>
        ` : ''}
      `;

      tooltip
        .html(tooltipContent)
        .style("visibility", "visible")
        .style("left", `${event.pageX + 10}px`)
        .style("top", `${event.pageY - 10}px`);
    };

    const hideTooltip = () => {
      tooltip.style("visibility", "hidden");
    };

    // Add interactions
    nodes
      .on("mouseover", (event, d) => {
        setHoveredNode(d);
        showTooltip(event, d);
        highlightPath(d);
      })
      .on("mousemove", (event, d) => {
        showTooltip(event, d);
      })
      .on("mouseout", () => {
        setHoveredNode(null);
        hideTooltip();
        resetHighlighting();
      })
      .on("click", (event, d) => {
        setSelectedPath(getPathToRoot(d));
        event.stopPropagation();
      });

    function getPathToRoot(node: d3.HierarchyNode<TreeNode>) {
      const path = [];
      let current = node;
      while (current.parent) {
        path.push(current);
        current = current.parent;
      }
      path.push(current);
      return path.reverse();
    }

    function highlightPath(node: d3.HierarchyNode<TreeNode>) {
      const path = getPathToRoot(node);
      
      // Dim all nodes and links
      svg.selectAll(".link")
        .style("stroke", "#eee")
        .style("stroke-width", "1px");
      
      svg.selectAll(".node circle")
        .style("opacity", 0.3);
      
      svg.selectAll(".node text")
        .style("opacity", 0.3);

      // Highlight the path
      path.forEach((node, i) => {
        if (i < path.length - 1) {
          const nextNode = path[i + 1];
          svg.selectAll(".link")
            .filter(d => {
              const linkData: any = d;
              return (linkData.source === node && linkData.target === nextNode);
            })
            .style("stroke", "#4f46e5")
            .style("stroke-width", "3px");
        }

        svg.selectAll(".node")
          .filter(d => d === node)
          .selectAll("circle, text")
          .style("opacity", 1);
      });
    }

    function resetHighlighting() {
      svg.selectAll(".link")
        .style("stroke", "#ccc")
        .style("stroke-width", "2px");
      
      svg.selectAll(".node circle, .node text")
        .style("opacity", 1);
    }

  }, [treeData, loading, dimensions]);

  // Add window resize handler
  useEffect(() => {
    const handleResize = () => {
      const container = svgRef.current?.parentElement;
      if (container) {
        setDimensions({
          width: Math.max(container.clientWidth, 1200),
          height: Math.max(container.clientHeight, 800)
        });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => window.removeEventListener('resize', handleResize);
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
                    value={feature.value}
                    onChange={(e) => handleFeatureChange(feature.name, Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-gray-500">
                    <span>{feature.min}</span>
                    <span>{feature.value}</span>
                    <span>{feature.max}</span>
                  </div>
                </div>
              ) : (
                <select
                  value={feature.value}
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