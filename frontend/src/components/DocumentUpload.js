import React, { useCallback, useState, useRef, useEffect } from "react";
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
  AppBar,
  Toolbar,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
} from "@mui/material";
import {
  CloudUpload,
  InsertDriveFile,
  Logout,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Cancel as CancelIcon,
  Info as InfoIcon,
} from "@mui/icons-material";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "react-pdf-highlighter/dist/style.css";
import { supabase } from "../supabaseclient";
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
  const [structuredAnalysis, setStructuredAnalysis] = useState([]); // table data
  const [timeline, setTimeline] = useState([]); // timeline data
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

  const renderStatusIcon = (status) => {
    if (status === "pass")
      return (
        <Tooltip title="Pass">
          <CheckIcon style={{ color: "green" }} />
        </Tooltip>
      );
    if (status === "fail")
      return (
        <Tooltip title="Fail">
          <CancelIcon style={{ color: "red" }} />
        </Tooltip>
      );
    if (status === "partial")
      return (
        <Tooltip title="Partial">
          <ErrorIcon style={{ color: "orange" }} />
        </Tooltip>
      );
    return <InfoIcon />;
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
  });

  useEffect(() => {
    if (analysisResult && analysisResult.highlightedPdf) {
      console.log("analysisResult:", analysisResult);
      console.log(analysisResult.highlightedPdf);
      const pdfUrl = URL.createObjectURL(analysisResult.highlightedPdf);

      setAnalysisResult({ ...analysisResult, highlightedPdf: pdfUrl });
    }
  }, [analysisResult]);
  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      window.location.href = "/auth";
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  const renderPriority = (priority) => {
    const style = {
      borderRadius: "20px",
      padding: "4px 12px",
      fontSize: "0.75rem",
      fontWeight: 600,
      display: "inline-block",
      textTransform: "capitalize",
    };

    if (priority === "high") {
      return (
        <span
          style={{ ...style, backgroundColor: "#fdecea", color: "#d32f2f" }}
        >
          High
        </span>
      );
    }
    if (priority === "medium") {
      return (
        <span
          style={{ ...style, backgroundColor: "#fff4e5", color: "#ed6c02" }}
        >
          Medium
        </span>
      );
    }
    if (priority === "low") {
      return (
        <span
          style={{ ...style, backgroundColor: "#edf7ed", color: "#2e7d32" }}
        >
          Low
        </span>
      );
    }
    return <span style={style}>{priority}</span>;
  };

  const handleAnalyze = async () => {
    if (files.length === 0) {
      alert("Please select at least one file first.");
      return;
    }

    // Clear all previous information
    setAnalysisResult(null);
    setStructuredAnalysis([]);
    setTimeline([]);
    setFileUrl("");
    setUploadMessage("");

    setLoading(true);
    setUploadMessage("");

    // Get current user from Supabase
    const {
      data: { user },
      error,
    } = await supabase.auth.getUser();

    if (error || !user) {
      alert("Could not get user info. Please log in again.");
      setLoading(false);
      return;
    }

    const userId = user.id;

    const formData = new FormData();
    formData.append("file", files[0].file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/analyze/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            "x-user-id": userId, // 🔥 pass user ID to backend
          },
        }
      );

      setAnalysisResult(response.data.analysis);
      setStructuredAnalysis(response.data.structured_analysis);
      console.log("Structured Analysis", response.data.structured_analysis);
      setTimeline(response.data.timeline);
      console.log("Timeline", response.data.timeline);
      console.log("response from api", response);
      const filePath = `http://127.0.0.1:8000/uploads/${response.data.highlighted_pdf_path
        .split("/")
        .pop()}`;
      setFileUrl(filePath);
      console.log(fileUrl);
      setUploadMessage(`Analysis complete for: ${response.data.filename}`);
      console.log("response from api", response.data);
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

          {/* Display the Structured Analysis */}
          {structuredAnalysis.length > 0 && (
            <TableContainer component={Paper} style={{ marginTop: 32 }}>
              <Typography variant="h6" style={{ padding: 16 }}>
                📊 Section-by-Section Compliance Summary
              </Typography>
              <Table>
                <TableHead>
                  <TableRow style={{ backgroundColor: "#f9f9f9" }}>
                    <TableCell>
                      <strong>Section</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Status</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Summary</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Rule</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Priority</strong>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {structuredAnalysis.map((row, idx) => (
                    <TableRow key={idx} hover>
                      <TableCell>{row.section}</TableCell>
                      <TableCell>{renderStatusIcon(row.status)}</TableCell>
                      <TableCell style={{ maxWidth: 400 }}>
                        {row.summary}
                      </TableCell>
                      <TableCell>{row.rule}</TableCell>
                      <TableCell>{renderPriority(row.priority)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {timeline.length > 0 && (
            <TableContainer component={Paper} style={{ marginTop: 48 }}>
              <Typography variant="h6" style={{ padding: 16 }}>
                📈 Change Timeline
              </Typography>
              <Table>
                <TableHead>
                  <TableRow style={{ backgroundColor: "#f9f9f9" }}>
                    <TableCell>
                      <strong>Section</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Previous</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Current</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Change Summary</strong>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {timeline.map((row, idx) => (
                    <TableRow key={idx} hover>
                      <TableCell>{row.section}</TableCell>
                      <TableCell>
                        {renderStatusIcon(row.previous_status)}
                      </TableCell>
                      <TableCell>
                        {renderStatusIcon(row.current_status)}
                      </TableCell>
                      <TableCell style={{ maxWidth: 500 }}>
                        {row.change_summary}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {/* Display the Structured Analysis */}
          {structuredAnalysis.length > 0 && (
            <TableContainer component={Paper} style={{ marginTop: 32 }}>
              <Typography variant="h6" style={{ padding: 16 }}>
                📊 Section-by-Section Compliance Summary
              </Typography>
              <Table>
                <TableHead>
                  <TableRow style={{ backgroundColor: "#f9f9f9" }}>
                    <TableCell>
                      <strong>Section</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Status</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Summary</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Rule</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Priority</strong>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {structuredAnalysis.map((row, idx) => (
                    <TableRow key={idx} hover>
                      <TableCell>{row.section}</TableCell>
                      <TableCell>{renderStatusIcon(row.status)}</TableCell>
                      <TableCell style={{ maxWidth: 400 }}>
                        {row.summary}
                      </TableCell>
                      <TableCell>{row.rule}</TableCell>
                      <TableCell>{renderPriority(row.priority)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {timeline.length > 0 && (
            <TableContainer component={Paper} style={{ marginTop: 48 }}>
              <Typography variant="h6" style={{ padding: 16 }}>
                📈 Change Timeline
              </Typography>
              <Table>
                <TableHead>
                  <TableRow style={{ backgroundColor: "#f9f9f9" }}>
                    <TableCell>
                      <strong>Section</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Previous</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Current</strong>
                    </TableCell>
                    <TableCell>
                      <strong>Change Summary</strong>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {timeline.map((row, idx) => (
                    <TableRow key={idx} hover>
                      <TableCell>{row.section}</TableCell>
                      <TableCell>
                        {renderStatusIcon(row.previous_status)}
                      </TableCell>
                      <TableCell>
                        {renderStatusIcon(row.current_status)}
                      </TableCell>
                      <TableCell style={{ maxWidth: 500 }}>
                        {row.change_summary}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {/* Display the Highlighted PDF */}
          {analysisResult && (
            <Paper elevation={2} style={{ padding: 24, marginTop: 32 }}>
              <Typography variant="h5" gutterBottom>
                Highlighted PDF
              </Typography>
              <embed
                src={fileUrl}
                type="application/pdf"
                width="100%"
                height="500"
              />
            </Paper>
          )}
        </Paper>
      </Container>
    </>
  );
};

export default DocumentUpload;
