import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.RFEA_BACKEND ?? 'http://localhost:8000'

// The dev server proxies the REST API and the WebSocket to the backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': backend,
      '/ws': { target: backend.replace(/^http/, 'ws'), ws: true },
    },
  },
})
