import React, { useCallback, useState, useRef, useEffect} from "react";
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
  Box,
} from "@mui/material";
import { CloudUpload, InsertDriveFile } from "@mui/icons-material";
import axios from "axios";
import ReactMarkdown from "react-markdown";
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
  const [analysisResult, setAnalysisResult] = useState(null);
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

  useEffect(() => {
    if (analysisResult && analysisResult.highlightedPdf) {
      console.log('analysisResult:', analysisResult);
      const pdfUrl = URL.createObjectURL(analysisResult.highlightedPdf);
      setAnalysisResult({...analysisResult, highlightedPdf: pdfUrl});
    }
  }, [analysisResult]);

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

      const highlightedPdfUrl = `http://127.0.0.1:8000/uploads/${response.data.highlighted_pdf_path.split('/').pop()}`;



      setAnalysisResult({...response.data,
        highlightedPdfUrl: highlightedPdfUrl
      });
      console.log("response from api", response.data);
      setFileUrl(response.data.file_url || "");
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

        {/* Display the Analysis Result */}
        {analysisResult && analysisResult.analysis && (
          <Paper elevation={2} style={{ padding: 24, marginTop: 32 }}>
            <Typography variant="h5" gutterBottom>
              Compliance Analysis
            </Typography>
            <Box mt={2}>
              <ReactMarkdown>{analysisResult.analysis.content}</ReactMarkdown>
            </Box>
          </Paper>
        )}

        {/* Display the Highlighted PDF */}
        {analysisResult && analysisResult.highlightedPdfUrl && (
          <Paper elevation={2} style={{ padding: 24, marginTop: 32}}>
            <Typography variant="h5" gutterBottom>
              Highlighted PDF
            </Typography>
            <embed
              src={analysisResult.highlightedPdfUrl}
              type="application/pdf"
              width="100%"
              height="500"
            />
          </Paper>
        )}

      </Paper>
    </Container>
  );
};

export default DocumentUpload;
