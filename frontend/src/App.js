// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DocumentUpload from './components/DocumentUpload';
import Auth from './components/Auth';
import ProtectedRoute from './components/protectedRoute';
import { CssBaseline } from '@mui/material';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/auth" element={<Auth />} />
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute>
              <CssBaseline />
              <DocumentUpload />
            </ProtectedRoute>
          } 
        />
        {/* Redirect root to dashboard or auth */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Router>
  );
}

export default App;