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
  Box,
} from "@mui/material";
import { CloudUpload, InsertDriveFile } from "@mui/icons-material";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "react-pdf-highlighter/dist/style.css";
import { supabase } from "../supabaseclient";
import { AppBar, Toolbar, IconButton } from "@mui/material";
import { Logout } from "@mui/icons-material";

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

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      window.location.href = "/auth";
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

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

      setAnalysisResult(response.data.analysis);
      console.log("response from api", response.data);
      setFileUrl(response.data.file_url || "");
      console.log("file_url", response.data.file_url);
      setUploadMessage(`Analysis complete for: ${response.data.filename}`);
    } catch (error) {
      console.error("Error analyzing file:", error);
      setUploadMessage("Error analyzing file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Legal Document Analyzer
          </Typography>
          <IconButton color="inherit" onClick={handleLogout} title="Logout">
            <Logout />
          </IconButton>
        </Toolbar>
      </AppBar>

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
                {loading ? (
                  <CircularProgress size={24} />
                ) : (
                  "Analyze Compliance"
                )}
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
          {analysisResult && (
            <Paper elevation={2} style={{ padding: 24, marginTop: 32 }}>
              <Typography variant="h5" gutterBottom>
                Compliance Analysis
              </Typography>
              <Box mt={2}>
                <ReactMarkdown>{analysisResult}</ReactMarkdown>
              </Box>
            </Paper>
          )}

          {/* Display PDF Viewer if a fileUrl exists */}
          {fileUrl && (
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
                      // Assuming you may add highlights later
                      highlights={[]}
                      highlightTransform={(
                        highlight,
                        index,
                        setTip,
                        hideTip
                      ) => (
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
    </>
  );
};

export default DocumentUpload;
