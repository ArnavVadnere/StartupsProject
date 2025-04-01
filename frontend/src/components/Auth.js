import React, { useState } from "react";
import { supabase } from "../supabaseclient";
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
  Grid,
  Divider,
  InputAdornment,
  IconButton,
} from "@mui/material";
import EmailIcon from "@mui/icons-material/Email";
import LockIcon from "@mui/icons-material/Lock";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import GavelIcon from "@mui/icons-material/Gavel";
import DescriptionIcon from "@mui/icons-material/Description";

function Auth() {
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [tab, setTab] = useState(0); // 0 for login, 1 for signup
  const [message, setMessage] = useState({ text: "", type: "" });

  const handleTabChange = (event, newValue) => {
    setTab(newValue);
    setMessage({ text: "", type: "" });
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage({ text: "", type: "" });

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) throw error;

      setMessage({ text: "Successfully logged in!", type: "success" });
      // Redirect to main app after successful login
      window.location.href = "/dashboard";
    } catch (error) {
      setMessage({ text: error.message, type: "error" });
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage({ text: "", type: "" });

    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });

      if (error) throw error;

      setMessage({
        text: "Registration successful! Please check your email for verification.",
        type: "success",
      });
    } catch (error) {
      setMessage({ text: error.message, type: "error" });
    } finally {
      setLoading(false);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <Container
      maxWidth="sm"
      sx={{
        display: "flex",
        alignItems: "center",
        minHeight: "100vh",
        py: 4,
      }}
    >
      <Paper
        elevation={4}
        sx={{
          width: "100%",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        {/* Header Section */}
        <Box
          sx={{
            bgcolor: "primary.main",
            color: "primary.contrastText",
            p: 3,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 1,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <GavelIcon fontSize="large" />
            <DescriptionIcon fontSize="large" />
          </Box>
          <Typography variant="h4" fontWeight="bold">
            Legal Compliance AI
          </Typography>
          <Typography variant="subtitle1">
            Automated document review and analysis
          </Typography>
        </Box>

        {/* Form Section */}
        <Box sx={{ p: 4 }}>
          <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
            <Tabs
              value={tab}
              onChange={handleTabChange}
              variant="fullWidth"
              sx={{
                "& .MuiTab-root": {
                  fontWeight: "bold",
                  py: 2,
                },
              }}
            >
              <Tab label="Login" />
              <Tab label="Sign Up" />
            </Tabs>
          </Box>

          {message.text && (
            <Alert
              severity={message.type}
              sx={{
                mb: 3,
                borderRadius: 1,
              }}
              onClose={() => setMessage({ text: "", type: "" })}
              variant="filled"
            >
              {message.text}
            </Alert>
          )}

          <Box
            component="form"
            onSubmit={tab === 0 ? handleLogin : handleSignUp}
          >
            <TextField
              label="Email Address"
              type="email"
              fullWidth
              margin="normal"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              variant="outlined"
              sx={{ mb: 2 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <EmailIcon color="primary" />
                  </InputAdornment>
                ),
              }}
            />
            <TextField
              label="Password"
              type={showPassword ? "text" : "password"}
              fullWidth
              margin="normal"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              variant="outlined"
              sx={{ mb: 3 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <LockIcon color="primary" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton onClick={togglePasswordVisibility} edge="end">
                      {showPassword ? (
                        <VisibilityOffIcon />
                      ) : (
                        <VisibilityIcon />
                      )}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />

            {tab === 0 && (
              <Box sx={{ textAlign: "right", mb: 2 }}>
                <Button
                  size="small"
                  sx={{
                    textTransform: "none",
                    fontWeight: "medium",
                  }}
                >
                  Forgot password?
                </Button>
              </Box>
            )}

            <Button
              type="submit"
              variant="contained"
              size="large"
              fullWidth
              sx={{
                mt: 1,
                py: 1.5,
                fontWeight: "bold",
                borderRadius: 2,
              }}
              disabled={loading}
            >
              {loading ? (
                <CircularProgress size={24} color="inherit" />
              ) : tab === 0 ? (
                "Sign In"
              ) : (
                "Create Account"
              )}
            </Button>

            {tab === 1 && (
              <Typography
                variant="caption"
                sx={{ mt: 2, display: "block", textAlign: "center" }}
              >
                By signing up, you agree to our Terms of Service and Privacy
                Policy
              </Typography>
            )}
          </Box>

          <Box sx={{ mt: 4, textAlign: "center" }}>
            <Divider sx={{ my: 2 }}>
              <Typography variant="body2" color="text.secondary">
                OR
              </Typography>
            </Divider>

            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Button
                  fullWidth
                  variant="outlined"
                  size="large"
                  sx={{
                    borderRadius: 2,
                    py: 1,
                  }}
                >
                  Continue with SSO
                </Button>
              </Grid>
            </Grid>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
}

export default Auth;
