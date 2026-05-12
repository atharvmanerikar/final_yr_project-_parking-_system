import express from 'express'
import path from 'path'
import { fileURLToPath } from 'url'
import { createProxyMiddleware } from 'http-proxy-middleware'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const PORT = process.env.PORT || 3000
const FLASK_TARGET = process.env.FLASK_TARGET || 'http://127.0.0.1:5000'

const app = express()

// Proxy all backend calls through Node
app.use(
  '/api',
  createProxyMiddleware({
    target: FLASK_TARGET,
    changeOrigin: true,
    ws: false,
    pathRewrite: {
      '^/api': '',
    },
    // Important for streaming endpoints like /live_feed
    selfHandleResponse: false,
    proxyTimeout: 0,
    timeout: 0,
  })
)

// Serve React build output
const distDir = path.join(__dirname, 'frontend', 'dist')
app.use(express.static(distDir))

// SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(distDir, 'index.html'))
})

app.listen(PORT, () => {
  console.log(`Node proxy server running: http://localhost:${PORT}`)
  console.log(`Proxying /api/* -> ${FLASK_TARGET}`)
})
