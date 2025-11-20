import React, { useState, useRef } from 'react';
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
  Menu,
  Close,
  Home,
  Dashboard,
  Explore,
  InfoOutlined,
} from '@mui/icons-material';
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from 'framer-motion';

// Magnification effect for navigation items
const NavItemWithMagnification: React.FC<{
  item: { label: string; path: string; icon: React.ReactNode };
  isActive: boolean;
  onClick: () => void;
  mouseX: any;
}> = ({ item, isActive, onClick, mouseX }) => {
  const ref = useRef<HTMLButtonElement>(null);

  const distance = useTransform(mouseX, (val: number) => {
    const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return val - bounds.x - bounds.width / 2;
  });

  const widthSync = useTransform(distance, [-150, 0, 150], [1, 1.08, 1]);
  const width = useSpring(widthSync, { mass: 0.1, stiffness: 150, damping: 12 });

  return (
    <motion.div style={{ scale: width }}>
      <Button
        ref={ref}
        onClick={onClick}
        startIcon={item.icon}
        sx={{
          color: isActive ? 'hsl(0, 0%, 100%)' : 'hsl(0, 0%, 75%)',
          textTransform: 'capitalize',
          fontFamily: '"Poppins", "Roboto", "Helvetica", sans-serif',
          fontSize: '0.95rem',
          letterSpacing: '0.02em',
          fontWeight: isActive ? 600 : 500,
          px: 2.5,
          py: 1,
          borderRadius: '25px',
          position: 'relative',
          transition: 'all 0.3s ease',
          background: isActive ? 'hsla(0, 0%, 100%, 0.15)' : 'transparent',
          border: isActive ? '1px solid hsla(0, 0%, 100%, 0.25)' : '1px solid transparent',
          '&:hover': {
            color: 'hsl(0, 0%, 100%)',
            background: 'hsla(0, 0%, 100%, 0.1)',
            border: '1px solid hsla(0, 0%, 100%, 0.2)',
            transform: 'translateY(-2px)',
          },
        }}
      >
        {item.label}
      </Button>
    </motion.div>
  );
};

const Navigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const mouseX = useMotionValue(Infinity);

  const navItems = [
    { label: 'Home', path: '/', icon: <Home /> },
    { label: 'Analyze', path: '/analyze', icon: <Dashboard /> },
    { label: 'Features', path: '/features', icon: <Explore /> },
    { label: 'About', path: '/about', icon: <InfoOutlined /> },
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
        background: 'hsla(0, 0%, 10%, 0.92)',
        backdropFilter: 'blur(40px)',
        WebkitBackdropFilter: 'blur(40px)',
        borderRight: '1px solid hsla(0, 0%, 100%, 0.15)',
      }}
    >
      <Box
        sx={{
          p: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid hsla(0, 0%, 100%, 0.1)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            component="img"
            src="https://vucvdpamtrjkzmubwlts.supabase.co/storage/v1/object/public/users/user_2rQ1QHrJyxpmWMHhqhANzWMc64n/avatar.png"
            alt="Logo"
            sx={{
              width: 30,
              height: 30,
              borderRadius: '50%',
              objectFit: 'cover'
            }}
          />
          <Typography
            variant="h6"
            sx={{
              color: 'hsl(0, 0%, 95%)',
              fontFamily: '"Poppins", "Roboto", "Helvetica", sans-serif',
              fontWeight: 700,
              fontSize: '1.1rem',
              letterSpacing: '0.05em',
            }}
          >
            DFS
          </Typography>
        </Box>
        <IconButton
          onClick={() => setMobileOpen(false)}
          sx={{
            color: 'hsl(0, 0%, 95%)',
            '&:hover': {
              backgroundColor: 'hsla(0, 0%, 100%, 0.1)',
            },
          }}
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
                borderRadius: '16px',
                py: 2,
                fontFamily: '"Poppins", "Roboto", "Helvetica", sans-serif',
                background: isActive(item.path)
                  ? 'hsla(0, 0%, 100%, 0.12)'
                  : 'transparent',
                border: isActive(item.path)
                  ? '1px solid hsla(0, 0%, 100%, 0.25)'
                  : '1px solid transparent',
                transition: 'all 0.3s ease',
                '&:hover': {
                  background: 'hsla(0, 0%, 100%, 0.08)',
                  borderColor: 'hsla(0, 0%, 100%, 0.2)',
                  transform: 'translateX(5px)',
                },
              }}
            >
              <Box sx={{
                color: isActive(item.path) ? 'hsl(0, 0%, 100%)' : 'hsl(0, 0%, 75%)',
                mr: 2,
                transition: 'all 0.3s ease',
              }}>
                {item.icon}
              </Box>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  fontFamily: '"Poppins", "Roboto", "Helvetica", sans-serif',
                  fontWeight: isActive(item.path) ? 600 : 500,
                  fontSize: '1rem',
                  letterSpacing: '0.02em',
                  color: isActive(item.path) ? 'hsl(0, 0%, 100%)' : 'hsl(0, 0%, 85%)',
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
        position="fixed"
        elevation={0}
        sx={{
          background: 'transparent',
          borderBottom: 'none',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          pt: 2,
          top: 0,
        }}
      >
        <motion.div
          initial={{ scaleX: 0, opacity: 0 }}
          animate={{ scaleX: 1, opacity: 1 }}
          transition={{
            duration: 0.6,
            ease: [0.4, 0, 0.2, 1],
            opacity: { duration: 0.4 }
          }}
          style={{
            width: '100%',
            display: 'flex',
            justifyContent: 'center',
          }}
        >
        <Toolbar
          sx={{
            py: 1.8,
            px: { xs: 2, md: 4 },
            width: { xs: '95%', md: '75%' },
            background: 'hsla(0, 0%, 100%, 0.08)',
            backdropFilter: 'blur(80px) saturate(150%)',
            WebkitBackdropFilter: 'blur(80px) saturate(150%)',
            borderRadius: '50px',
            border: '1px solid hsla(0, 0%, 100%, 0.15)',
            boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 0 hsla(0, 0%, 100%, 0.1)',
            transition: 'all 0.3s ease-in-out',
            '&:hover': {
              background: 'hsla(0, 0%, 100%, 0.12)',
              border: '1px solid hsla(0, 0%, 100%, 0.2)',
              boxShadow: '0 12px 40px 0 rgba(0, 0, 0, 0.4), inset 0 1px 0 hsla(0, 0%, 100%, 0.15)',
            },
          }}
        >
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
            <Box
              component="img"
              src="https://vucvdpamtrjkzmubwlts.supabase.co/storage/v1/object/public/users/user_2rQ1QHrJyxpmWMHhqhANzWMc64n/avatar.png"
              alt="Logo"
              sx={{
                width: { xs: 28, md: 32 },
                height: { xs: 28, md: 32 },
                borderRadius: '50%',
                objectFit: 'cover',
                filter: 'drop-shadow(0 0 8px hsla(0, 0%, 100%, 0.3))',
                transition: 'all 0.3s ease',
                '&:hover': {
                  filter: 'drop-shadow(0 0 12px hsla(0, 0%, 100%, 0.5))',
                  transform: 'scale(1.05) rotate(-5deg)',
                },
              }}
            />
            <Typography
              variant="h6"
              sx={{
                color: 'hsl(0, 0%, 95%)',
                fontFamily: '"Poppins", "Roboto", "Helvetica", sans-serif',
                fontWeight: 700,
                fontSize: { xs: '0.9rem', md: '1.05rem' },
                letterSpacing: '0.05em',
                display: { xs: 'none', sm: 'block' },
              }}
            >
              DEEPFAKE DETECTION
            </Typography>
          </Box>

          <Box sx={{ flexGrow: 1 }} />

          {/* Desktop Navigation */}
          {!isMobile && (
            <Box
              sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
              onMouseMove={(e) => mouseX.set(e.pageX)}
              onMouseLeave={() => mouseX.set(Infinity)}
            >
              {navItems.map((item) => (
                <NavItemWithMagnification
                  key={item.path}
                  item={item}
                  isActive={isActive(item.path)}
                  onClick={() => handleNavClick(item.path)}
                  mouseX={mouseX}
                />
              ))}
              <Chip
                label="AI-POWERED"
                size="small"
                sx={{
                  ml: 2,
                  background: 'hsla(0, 0%, 100%, 0.12)',
                  border: '1px solid hsla(0, 0%, 100%, 0.3)',
                  color: 'hsl(0, 0%, 95%)',
                  fontFamily: '"Poppins", "Roboto", "Helvetica", sans-serif',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    background: 'hsla(0, 0%, 100%, 0.18)',
                    transform: 'scale(1.05)',
                  },
                }}
              />
            </Box>
          )}

          {/* Mobile Menu Button */}
          {isMobile && (
            <IconButton
              onClick={() => setMobileOpen(true)}
              sx={{
                color: 'hsl(0, 0%, 95%)',
                border: '1px solid hsla(0, 0%, 100%, 0.25)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                '&:hover': {
                  borderColor: 'hsla(0, 0%, 100%, 0.4)',
                  backgroundColor: 'hsla(0, 0%, 100%, 0.1)',
                  transform: 'scale(1.05)',
                },
              }}
            >
              <Menu />
            </IconButton>
          )}
        </Toolbar>
        </motion.div>
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
