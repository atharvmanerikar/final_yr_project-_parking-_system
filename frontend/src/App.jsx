import { useState, useEffect } from 'react'
import './index.css'

function App() {
  const [totalSlots, setTotalSlots] = useState(0)
  const [occupiedSlots, setOccupiedSlots] = useState(0)
  const [freeSlots, setFreeSlots] = useState(0)
  const [capacityUsage, setCapacityUsage] = useState(0)
  const [isLive, setIsLive] = useState(false)
  const [liveFeedUrl, setLiveFeedUrl] = useState('')
  const [feedKey, setFeedKey] = useState(0)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/status')
        if (res.ok) {
          const data = await res.json()
          setTotalSlots(data.total_slots || 0)
          setOccupiedSlots(data.occupied || 0)
          setFreeSlots(data.free || 0)
          setCapacityUsage(data.occupancy_rate || 0)
          setError('')
        }
      } catch (e) {
        setError('Backend not running')
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!isLive) return
    const interval = setInterval(() => {
      setFeedKey(k => k + 1)
    }, 3000)
    return () => clearInterval(interval)
  }, [isLive])

  const startVideo = async () => {
    setError('')
    console.log('[startVideo] Starting video...')
    try {
      const res = await fetch('/api/start_video_file', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          path: 'c:\\\\Users\\\\athar\\\\Desktop\\\\final yr proj\\\\WhatsApp Video 2026-04-22 at 9.56.53 PM.mp4'
        })
      })
      
      if (res.ok) {
        console.log('[startVideo] Backend started successfully')
        setIsLive(true)
        setLiveFeedUrl('/api/live_feed')
        setFeedKey(k => k + 1)
      } else {
        const err = await res.json().catch(() => ({}))
        if (res.status === 409 && (err.error || '').toLowerCase().includes('already')) {
          console.log('[startVideo] Already running, switching to live')
          setIsLive(true)
          setLiveFeedUrl('/api/live_feed')
          setFeedKey(k => k + 1)
        } else {
          console.log('[startVideo] Error:', err)
          setError(err.error || 'Failed to start')
        }
      }
    } catch (e) {
      console.log('[startVideo] Network error:', e)
      setError('Network error')
    }
  }

  const stopVideo = async () => {
    try {
      await fetch('/api/stop_camera', {method: 'POST'})
      setIsLive(false)
      setLiveFeedUrl('')
      setFeedKey(0)
    } catch (e) {
      setError('Error stopping')
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1>🚗 Smart Parking Dashboard</h1>
          <p>Real-time parking monitoring and analytics</p>
        </div>
        <div className="live-badge">
          <div className="live-dot"></div>
          LIVE
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      <section className="stats">
        <div className="card">
          <div className="card-icon ic-blue">🅿️</div>
          <div className="card-info">
            <label>Total Slots</label>
            <div className="val val-blue">{totalSlots}</div>
          </div>
        </div>
        <div className="card">
          <div className="card-icon ic-red">🚙</div>
          <div className="card-info">
            <label>Occupied</label>
            <div className="val val-red">{occupiedSlots}</div>
          </div>
        </div>
        <div className="card">
          <div className="card-icon ic-green">✅</div>
          <div className="card-info">
            <label>Available</label>
            <div className="val val-green">{freeSlots}</div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <span className="panel-title">Capacity Overview</span>
          <span className="pct-label">{capacityUsage}%</span>
        </div>
        <div className="sub-text">{occupiedSlots} of {totalSlots} slots occupied</div>
        <div className="progress-bar-bg">
          <div 
            className="progress-bar-fill" 
            style={{ 
              width: `${capacityUsage}%`,
              background: capacityUsage > 80 ? '#e53935' : capacityUsage > 50 ? '#fb8c00' : '#43a047'
            }}
          ></div>
        </div>
      </section>

      <section className="panel">
        <div className="cam-header">
          <span className="panel-title">Live Camera Feed</span>
          <div className="cam-actions">
            {!isLive ? (
              <button className="btn-stop" onClick={startVideo}>
                ▶ Start
              </button>
            ) : (
              <button className="btn-stop" onClick={stopVideo}>
                ⏹ Stop
              </button>
            )}
            <a
              href="/api/live_feed"
              target="_blank"
              rel="noreferrer"
              className="btn-open"
            >
              🔗 Open Feed
            </a>
          </div>
        </div>
        <div className="cam-feed">
          {isLive && liveFeedUrl ? (
            <img 
              key={feedKey}
              src={`${liveFeedUrl}?t=${Date.now()}`} 
              alt="Live feed" 
              className="video-stream"
              onLoad={() => console.log('[Video] Live feed loaded')}
              onError={(e) => {
                console.log('[Video] Live feed failed, trying fallback')
                e.target.src = '/api/latest_result?t=' + Date.now()
              }}
            />
          ) : (
            <div className="cam-slots">
              <div className="cam-slot occ">
                <span style={{fontSize: '18px'}}>🚗</span>
                <span>Occupied</span>
                <span className="slot-num">Slot 1</span>
              </div>
              <div className="cam-slot occ">
                <span style={{fontSize: '18px'}}>🚗</span>
                <span>Occupied</span>
                <span className="slot-num">Slot 2</span>
              </div>
              <div className="cam-slot occ">
                <span style={{fontSize: '18px'}}>🚗</span>
                <span>Occupied</span>
                <span className="slot-num">Slot 3</span>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

export default App
