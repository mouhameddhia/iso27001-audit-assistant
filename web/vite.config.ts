import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Proxies /api/* to the FastAPI backend during development, so the frontend only ever talks to
// its own origin -- the same shape a production reverse proxy would use (nginx routing /api to
// the backend and everything else to the built static files), just without one yet.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
