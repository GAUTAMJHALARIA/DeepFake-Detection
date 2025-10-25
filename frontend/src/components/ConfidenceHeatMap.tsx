import React, { useRef, useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Slider,
  FormControlLabel,
  Switch,
} from '@mui/material';
import * as d3 from 'd3';

interface FrameData {
  index: number;
  timestamp: number;
  confidence: number;
  label: string;
  face_detected: boolean;
}

interface ConfidenceHeatMapProps {
  analysisData: {
    frames: FrameData[];
    video_info: {
      duration: number;
      fps: number;
    };
    statistics: {
      mean_confidence: number;
      max_confidence: number;
      min_confidence: number;
    };
  };
  currentFrameIndex: number;
  onFrameSelect: (frameIndex: number) => void;
}

const ConfidenceHeatMap: React.FC<ConfidenceHeatMapProps> = ({
  analysisData,
  currentFrameIndex,
  onFrameSelect,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showFaceDetection, setShowFaceDetection] = useState(true);
  const [heatMapHeight, setHeatMapHeight] = useState(200);

  useEffect(() => {
    if (!svgRef.current || !analysisData.frames.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const width = 800 - margin.left - margin.right;
    const height = heatMapHeight - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain([0, analysisData.video_info.duration])
      .range([0, width]);

    const yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Color scale for confidence (Red-Yellow-Green)
    const colorScale = d3
      .scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateRdYlGn)
      .clamp(true);

    // Create heat map rectangles
    const rectWidth = width / analysisData.frames.length;

    g.selectAll(".confidence-rect")
      .data(analysisData.frames)
      .enter()
      .append("rect")
      .attr("class", "confidence-rect")
      .attr("x", (d) => xScale(d.timestamp))
      .attr("y", 0)
      .attr("width", Math.max(1, rectWidth))
      .attr("height", height)
      .attr("fill", (d) => colorScale(1 - d.confidence)) // Invert for Red-Yellow-Green
      .attr("stroke", "none")
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        onFrameSelect(d.index);
      })
      .on("mouseover", function(event, d) {
        // Tooltip
        const tooltip = d3.select("body")
          .append("div")
          .attr("class", "tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0, 0, 0, 0.8)")
          .style("color", "white")
          .style("padding", "8px")
          .style("border-radius", "4px")
          .style("font-size", "12px")
          .style("pointer-events", "none")
          .style("z-index", 1000);

        tooltip
          .html(`
            <div>Frame ${d.index + 1}</div>
            <div>Time: ${d.timestamp.toFixed(2)}s</div>
            <div>Confidence: ${(d.confidence * 100).toFixed(1)}%</div>
            <div>Label: ${d.label}</div>
            <div>Face: ${d.face_detected ? 'Yes' : 'No'}</div>
          `)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px");

        d3.select(this).attr("stroke", "white").attr("stroke-width", 2);
      })
      .on("mouseout", function() {
        d3.selectAll(".tooltip").remove();
        d3.select(this).attr("stroke", "none");
      });

    // Face detection indicators
    if (showFaceDetection) {
      g.selectAll(".face-indicator")
        .data(analysisData.frames.filter(d => !d.face_detected))
        .enter()
        .append("circle")
        .attr("class", "face-indicator")
        .attr("cx", (d) => xScale(d.timestamp) + rectWidth / 2)
        .attr("cy", height + 15)
        .attr("r", 3)
        .attr("fill", "#ff5722")
        .style("cursor", "pointer")
        .on("click", (event, d) => {
          onFrameSelect(d.index);
        });
    }

    // Current frame indicator
    const currentFrame = analysisData.frames[currentFrameIndex];
    if (currentFrame) {
      g.append("line")
        .attr("class", "current-frame-line")
        .attr("x1", xScale(currentFrame.timestamp) + rectWidth / 2)
        .attr("x2", xScale(currentFrame.timestamp) + rectWidth / 2)
        .attr("y1", -10)
        .attr("y2", height + 30)
        .attr("stroke", "white")
        .attr("stroke-width", 3)
        .attr("stroke-dasharray", "5,5");

      g.append("circle")
        .attr("class", "current-frame-indicator")
        .attr("cx", xScale(currentFrame.timestamp) + rectWidth / 2)
        .attr("cy", -10)
        .attr("r", 6)
        .attr("fill", "white")
        .attr("stroke", "#1976d2")
        .attr("stroke-width", 2);
    }

    // Axes
    const xAxis = d3.axisBottom(xScale).tickFormat((d) => `${d}s`);
    const yAxis = d3.axisLeft(yScale).tickFormat((d) => `${((d as number) * 100).toFixed(0)}%`);

    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(xAxis)
      .append("text")
      .attr("x", width / 2)
      .attr("y", 35)
      .attr("fill", "currentColor")
      .style("text-anchor", "middle")
      .text("Time (seconds)");

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -height / 2)
      .attr("fill", "currentColor")
      .style("text-anchor", "middle")
      .text("Confidence (%)");

    // Legend
    const legendWidth = 200;
    const legendHeight = 20;
    const legend = g.append("g")
      .attr("transform", `translate(${width - legendWidth - 20}, 20)`);

    const legendScale = d3.scaleLinear()
      .domain([0, legendWidth])
      .range([0, 1]);

    legend.selectAll(".legend-rect")
      .data(d3.range(legendWidth))
      .enter()
      .append("rect")
      .attr("x", d => d)
      .attr("y", 0)
      .attr("width", 1)
      .attr("height", legendHeight)
      .attr("fill", d => colorScale(1 - legendScale(d)));

    legend.append("text")
      .attr("x", 0)
      .attr("y", legendHeight + 15)
      .attr("fill", "currentColor")
      .style("font-size", "12px")
      .text("Real");

    legend.append("text")
      .attr("x", legendWidth - 20)
      .attr("y", legendHeight + 15)
      .attr("fill", "currentColor")
      .style("font-size", "12px")
      .text("Fake");

  }, [analysisData, currentFrameIndex, showFaceDetection, heatMapHeight, zoomLevel]);

  const getConfidenceStats = () => {
    const { frames } = analysisData;
    const suspiciousFrames = frames.filter(f => f.confidence >= 0.6).length;
    const uncertainFrames = frames.filter(f => f.confidence >= 0.3 && f.confidence < 0.6).length;
    const authenticFrames = frames.filter(f => f.confidence < 0.3).length;

    return { suspiciousFrames, uncertainFrames, authenticFrames };
  };

  const stats = getConfidenceStats();

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Confidence Heat Map
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Interactive timeline showing confidence scores across the entire video. Click on any point to jump to that frame.
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" gap={3} alignItems="center" flexWrap="wrap">
            <Box>
              <Typography variant="body2" gutterBottom>
                Heat Map Height
              </Typography>
              <Slider
                value={heatMapHeight}
                onChange={(_, value) => setHeatMapHeight(value as number)}
                min={150}
                max={400}
                step={50}
                marks={[
                  { value: 150, label: '150px' },
                  { value: 200, label: '200px' },
                  { value: 300, label: '300px' },
                  { value: 400, label: '400px' },
                ]}
                sx={{ width: 200 }}
              />
            </Box>

            <FormControlLabel
              control={
                <Switch
                  checked={showFaceDetection}
                  onChange={(e) => setShowFaceDetection(e.target.checked)}
                />
              }
              label="Show Face Detection Issues"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Heat Map */}
      <Paper sx={{ p: 2, mb: 3, overflow: 'auto' }}>
        <svg
          ref={svgRef}
          width="100%"
          height={heatMapHeight}
          style={{ minWidth: 800 }}
        />
      </Paper>

      {/* Statistics */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Frame Analysis Summary
          </Typography>

          <Box display="flex" gap={4} flexWrap="wrap">
            <Box textAlign="center">
              <Typography variant="h4" color="error">
                {stats.suspiciousFrames}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Suspicious Frames
              </Typography>
              <Typography variant="caption" color="textSecondary">
                ≥60% confidence
              </Typography>
            </Box>

            <Box textAlign="center">
              <Typography variant="h4" color="warning.main">
                {stats.uncertainFrames}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Uncertain Frames
              </Typography>
              <Typography variant="caption" color="textSecondary">
                30-60% confidence
              </Typography>
            </Box>

            <Box textAlign="center">
              <Typography variant="h4" color="success.main">
                {stats.authenticFrames}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Authentic Frames
              </Typography>
              <Typography variant="caption" color="textSecondary">
                &lt;30% confidence
              </Typography>
            </Box>

            <Box textAlign="center">
              <Typography variant="h4" color="info.main">
                {analysisData.statistics.mean_confidence.toFixed(3)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Average Confidence
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Overall score
              </Typography>
            </Box>
          </Box>

          <Box mt={2}>
            <Typography variant="body2" color="textSecondary">
              <strong>Legend:</strong> Green = Authentic, Yellow = Uncertain, Red = Suspicious
            </Typography>
            <Typography variant="body2" color="textSecondary">
              <strong>Red dots below timeline:</strong> Frames without face detection
            </Typography>
            <Typography variant="body2" color="textSecondary">
              <strong>White dashed line:</strong> Current frame position
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ConfidenceHeatMap;
