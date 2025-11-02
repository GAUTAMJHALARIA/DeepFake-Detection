import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Chip,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  Security,
  Menu,
  Close,
  Home,
  Analytics,
  Science,
  Info,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

const Navigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);

  const navItems = [
    { label: 'Home', path: '/', icon: <Home /> },
    { label: 'Analyze', path: '/analyze', icon: <Analytics /> },
    { label: 'Features', path: '/features', icon: <Science /> },
    { label: 'About', path: '/about', icon: <Info /> },
  ];

  const isActive = (path: string) => location.pathname === path;

  const handleNavClick = (path: string) => {
    navigate(path);
    setMobileOpen(false);
  };

  const drawer = (
    <Box
      sx={{
        width: 280,
        height: '100%',
        background: 'linear-gradient(135deg, hsla(220, 35%, 10%, 0.98) 0%, hsla(220, 40%, 8%, 0.98) 100%)',
        backdropFilter: 'blur(20px)',
        borderRight: '1px solid hsla(200, 100%, 55%, 0.2)',
      }}
    >
      <Box
        sx={{
          p: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid hsla(200, 100%, 55%, 0.1)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Security sx={{ color: 'hsl(200, 100%, 55%)', fontSize: 28 }} />
          <Typography
            variant="h6"
            sx={{
              background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              fontWeight: 700,
              fontSize: '1rem',
              letterSpacing: '0.05em',
            }}
          >
            DFS
          </Typography>
        </Box>
        <IconButton
          onClick={() => setMobileOpen(false)}
          sx={{ color: 'hsl(200, 100%, 55%)' }}
        >
          <Close />
        </IconButton>
      </Box>

      <List sx={{ pt: 2 }}>
        {navItems.map((item, index) => (
          <ListItem key={item.path} disablePadding sx={{ mb: 1 }}>
            <ListItemButton
              onClick={() => handleNavClick(item.path)}
              sx={{
                mx: 2,
                borderRadius: '12px',
                py: 2,
                background: isActive(item.path)
                  ? 'linear-gradient(135deg, hsla(200, 100%, 55%, 0.15) 0%, hsla(200, 100%, 55%, 0.1) 100%)'
                  : 'transparent',
                border: isActive(item.path)
                  ? '1px solid hsla(200, 100%, 55%, 0.4)'
                  : '1px solid transparent',
                '&:hover': {
                  background: 'hsla(200, 100%, 55%, 0.1)',
                  borderColor: 'hsla(200, 100%, 55%, 0.3)',
                },
              }}
            >
              <Box sx={{ color: isActive(item.path) ? 'hsl(200, 100%, 55%)' : 'text.secondary', mr: 2 }}>
                {item.icon}
              </Box>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  fontWeight: isActive(item.path) ? 700 : 500,
                  fontSize: '1rem',
                  letterSpacing: '0.05em',
                  color: isActive(item.path) ? 'hsl(200, 100%, 55%)' : 'text.primary',
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <>
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          background: 'hsla(220, 35%, 10%, 0.85)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid hsla(200, 100%, 55%, 0.2)',
          zIndex: 1000,
        }}
      >
        <Toolbar sx={{ py: 1.5, px: { xs: 2, md: 4 } }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              cursor: 'pointer',
              flexGrow: { xs: 1, md: 0 },
            }}
            onClick={() => handleNavClick('/')}
          >
            <Security
              sx={{
                color: 'hsl(200, 100%, 55%)',
                filter: 'drop-shadow(0 0 8px hsl(200, 100%, 55%))',
                animation: 'pulse 3s infinite',
                fontSize: { xs: 24, md: 28 },
              }}
            />
            <Typography
              variant="h6"
              sx={{
                background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                fontWeight: 700,
                fontSize: { xs: '1rem', md: '1.25rem' },
                letterSpacing: '0.02em',
                display: { xs: 'none', sm: 'block' },
              }}
            >
              DEEPFAKE DETECTION SYSTEM
            </Typography>
          </Box>

          <Box sx={{ flexGrow: 1 }} />

          {/* Desktop Navigation */}
          {!isMobile && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {navItems.map((item) => (
                <Button
                  key={item.path}
                  onClick={() => handleNavClick(item.path)}
                  startIcon={item.icon}
                  sx={{
                    color: isActive(item.path)
                      ? 'hsl(200, 100%, 55%)'
                      : 'text.secondary',
                    textTransform: 'uppercase',
                    fontSize: '0.85rem',
                    letterSpacing: '0.1em',
                    fontWeight: isActive(item.path) ? 700 : 500,
                    px: 2,
                    py: 1,
                    borderRadius: '8px',
                    position: 'relative',
                    '&::after': {
                      content: '""',
                      position: 'absolute',
                      bottom: 0,
                      left: '50%',
                      transform: 'translateX(-50%)',
                      width: isActive(item.path) ? '80%' : '0%',
                      height: '2px',
                      background: 'hsl(200, 100%, 55%)',
                      transition: 'width 0.3s ease',
                      boxShadow: isActive(item.path) ? '0 0 10px hsl(200, 100%, 55%)' : 'none',
                    },
                    '&:hover': {
                      color: 'hsl(200, 100%, 55%)',
                      backgroundColor: 'hsla(200, 100%, 55%, 0.1)',
                    },
                  }}
                >
                  {item.label}
                </Button>
              ))}
              <Chip
                label="AI-POWERED"
                size="small"
                sx={{
                  ml: 2,
                  background: 'hsla(200, 100%, 55%, 0.15)',
                  border: '1px solid hsla(200, 100%, 55%, 0.4)',
                  color: 'hsl(200, 100%, 55%)',
                  fontWeight: 600,
                  fontSize: '0.7rem',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  boxShadow: '0 0 10px hsla(200, 100%, 55%, 0.2)',
                }}
              />
            </Box>
          )}

          {/* Mobile Menu Button */}
          {isMobile && (
            <IconButton
              onClick={() => setMobileOpen(true)}
              sx={{
                color: 'hsl(200, 100%, 55%)',
                border: '1px solid hsla(200, 100%, 55%, 0.3)',
                '&:hover': {
                  borderColor: 'hsl(200, 100%, 55%)',
                  backgroundColor: 'hsla(200, 100%, 55%, 0.1)',
                },
              }}
            >
              <Menu />
            </IconButton>
          )}
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <Drawer
        anchor="left"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        ModalProps={{
          keepMounted: true,
        }}
        sx={{
          '& .MuiDrawer-paper': {
            backgroundColor: 'transparent',
            border: 'none',
          },
        }}
      >
        <AnimatePresence>
          {mobileOpen && (
            <motion.div
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            >
              {drawer}
            </motion.div>
          )}
        </AnimatePresence>
      </Drawer>
    </>
  );
};

export default Navigation;
