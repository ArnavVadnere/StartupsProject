import React, { useCallback, useState, useRef } from "react";
import { useDropzone } from "react-dropzone";
import {
  Button,
  Container,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  CircularProgress,
} from "@mui/material";
import { CloudUpload, InsertDriveFile } from "@mui/icons-material";
import axios from "axios";
import "react-pdf-highlighter/dist/style.css";

import {
  PdfLoader,
  PdfHighlighter,
  Highlight,
  Popup,
} from "react-pdf-highlighter";

const DocumentUpload = () => {
  const [files, setFiles] = useState([]);
  const [fileUrl, setFileUrl] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [analysisResult, setAnalysisResult] = useState([]);
  const [loading, setLoading] = useState(false);
  const scrollViewerTo = useRef(() => {});

  const onDrop = useCallback((acceptedFiles) => {
    setFiles(
      acceptedFiles.map((file) => ({
        file,
        id: Date.now() + Math.random(),
        name: file.name,
        size: (file.size / 1024).toFixed(2) + " KB",
      }))
    );
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
  });

  const handleAnalyze = async () => {
    if (files.length === 0) {
      alert("Please select at least one file first.");
      return;
    }

    setLoading(true);
    setUploadMessage("");

    const formData = new FormData();
    formData.append("file", files[0].file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/analyze/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setAnalysisResult(response.data.compliance_issues);
      setFileUrl(response.data.file_url);
      setUploadMessage(`Analysis complete for: ${response.data.filename}`);
    } catch (error) {
      console.error("Error analyzing file:", error);
      setUploadMessage("Error analyzing file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" style={{ marginTop: 32 }}>
      <Paper elevation={3} style={{ padding: 32 }}>
        <Typography variant="h4" gutterBottom align="center">
          Legal Document Upload & Analysis
        </Typography>

        <div
          {...getRootProps()}
          style={{
            border: "2px dashed #1976d2",
            borderRadius: 8,
            padding: 32,
            textAlign: "center",
            cursor: "pointer",
            marginBottom: 32,
          }}
        >
          <input {...getInputProps()} />
          <CloudUpload
            style={{ fontSize: 60, color: "#1976d2", marginBottom: 16 }}
          />
          <Typography variant="body1">
            Drag & drop a legal document here, or click to select
          </Typography>
        </div>

        {files.length > 0 && (
          <>
            <Typography variant="h6" gutterBottom>
              Selected File:
            </Typography>
            <List>
              {files.map(({ id, name, size }) => (
                <ListItem key={id}>
                  <ListItemIcon>
                    <InsertDriveFile color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={`${name} (${size})`} />
                </ListItem>
              ))}
            </List>
            <Button
              variant="contained"
              size="large"
              fullWidth
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : "Analyze Compliance"}
            </Button>
          </>
        )}

        {uploadMessage && (
          <Typography
            variant="body2"
            color="green"
            align="center"
            style={{ marginTop: 16 }}
          >
            {uploadMessage}
          </Typography>
        )}

        {fileUrl && (
          // Outer container with relative positioning
          <div
            style={{
              position: "relative",
              height: "80vh",
              border: "1px solid #ccc",
              marginTop: 32,
            }}
          >
            <PdfLoader url={fileUrl} beforeLoad={<CircularProgress />}>
              {(pdfDocument) => (
                // Inner wrapper with absolute positioning
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                  }}
                >
                  <PdfHighlighter
                    pdfDocument={pdfDocument}
                    highlights={analysisResult.map((issue, i) => ({
                      id: `highlight-${i}`,
                      position: {
                        pageNumber: issue.page,
                        boundingRect: {
                          x1: issue.x1,
                          x0: issue.x0,
                          y1: issue.y1,
                          y0: issue.y0,
                        },
                        rects: [],
                      },
                      content: { text: issue.text },
                    }))}
                    highlightTransform={(highlight, index, setTip, hideTip) => (
                      <Popup
                        popupContent={<div>{highlight.content.text}</div>}
                        onMouseOver={setTip}
                        onMouseOut={hideTip}
                        key={index}
                      >
                        <Highlight
                          position={highlight.position}
                          comment={highlight.content.text}
                        />
                      </Popup>
                    )}
                    onSelectionFinished={() => {}}
                    enableAreaSelection={() => false}
                    // Also pass the style prop here for extra assurance.
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                    }}
                  />
                </div>
              )}
            </PdfLoader>
          </div>
        )}
      </Paper>
    </Container>
  );
};

export default DocumentUpload;
