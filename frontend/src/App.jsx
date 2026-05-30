import { useState, useEffect } from 'react'
import './index.css'
import parkingData from './assets/parking_slots.json'

function App() {
  const [totalSlots, setTotalSlots] = useState(0)
  const [occupiedSlots, setOccupiedSlots] = useState(0)
  const [freeSlots, setFreeSlots] = useState(0)
  const [capacityUsage, setCapacityUsage] = useState(0)
  const [isLive, setIsLive] = useState(false)
  const [liveFeedUrl, setLiveFeedUrl] = useState('')
  const [feedKey, setFeedKey] = useState(0)
  const [error, setError] = useState('')
  const [slotStatus, setSlotStatus] = useState({})
  const [pathCoords, setPathCoords] = useState([])
  const [bestSlot, setBestSlot] = useState("")
  const [activeTab, setActiveTab] = useState('live')
  const [trackingLogs, setTrackingLogs] = useState([])
  const [filterDate, setFilterDate] = useState('')
  const [filterTime, setFilterTime] = useState('')
  const parkingSlots = parkingData.slots

  const fetchPath = async () => {
    try {
      const response = await fetch('/api/path')
      if (response.ok) {
        const data = await response.json()
        setPathCoords(data.coords || [])
        setBestSlot(data.slot || "")
      }
    } catch (error) {
      console.error("Path Error:", error)
    }
  }

  const fetchTrackingLogs = async () => {
    try {
      const response = await fetch('/api/tracking')
      if (response.ok) {
        const data = await response.json()
        setTrackingLogs(data.tracks || [])
      }
    } catch (error) {
      console.error("Error fetching tracking logs:", error)
    }
  }

  useEffect(() => {
    if (activeTab === 'history') {
      fetchTrackingLogs()
    }
  }, [activeTab])

  const formatTime = (isoStr) => {
    if (!isoStr) return '-'
    try {
      const d = new Date(isoStr)
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    } catch (e) {
      return isoStr
    }
  }

  const formatDate = (dateStr) => {
    if (!dateStr || dateStr === 'Unknown Date') return 'Unknown Date'
    try {
      const d = new Date(dateStr)
      return d.toLocaleDateString([], { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })
    } catch (e) {
      return dateStr
    }
  }

  const matchesFilter = (log) => {
    let dateMatches = true
    let timeMatches = true

    const entryDate = log.entry_time ? log.entry_time.split('T')[0] : null
    const exitDate = log.exit_time ? log.exit_time.split('T')[0] : null
    
    if (filterDate) {
      dateMatches = entryDate === filterDate || exitDate === filterDate
    }

    if (filterTime) {
      const getMinutes = (tStr) => {
        if (!tStr) return null
        const parts = tStr.split('T')
        if (parts.length < 2) return null
        const tParts = parts[1].split(':')
        return parseInt(tParts[0], 10) * 60 + parseInt(tParts[1], 10)
      }

      const filterMin = parseInt(filterTime.split(':')[0], 10) * 60 + parseInt(filterTime.split(':')[1], 10)
      const entryMin = getMinutes(log.entry_time)
      const exitMin = getMinutes(log.exit_time)

      if (entryMin !== null) {
        if (exitMin !== null) {
          timeMatches = filterMin >= entryMin && filterMin <= exitMin
        } else {
          timeMatches = filterMin >= entryMin
        }
      } else {
        timeMatches = false
      }
    }

    return dateMatches && timeMatches
  }

  const filteredLogs = trackingLogs.filter(matchesFilter)

  const groupedLogs = {}
  filteredLogs.forEach(log => {
    const dateStr = log.entry_time ? log.entry_time.split('T')[0] : (log.exit_time ? log.exit_time.split('T')[0] : 'Unknown Date')
    if (!groupedLogs[dateStr]) {
      groupedLogs[dateStr] = []
    }
    groupedLogs[dateStr].push(log)
  })

  const sortedDates = Object.keys(groupedLogs).sort((a, b) => {
    if (a === 'Unknown Date') return 1
    if (b === 'Unknown Date') return -1
    return new Date(b) - new Date(a)
  })

  const drawPath = (coords) => {
    const canvas = document.getElementById("pathCanvas")
    if (!canvas || coords.length === 0) {
      const ctx = canvas?.getContext("2d")
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
      return
    }

    const ctx = canvas.getContext("2d")

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const originalWidth = 2950
    const originalHeight = 1440

    const scaleX = canvas.width / originalWidth
    const scaleY = canvas.height / originalHeight

    ctx.beginPath()
    ctx.moveTo(coords[0][0] * scaleX, coords[0][1] * scaleY)

    for (let i = 1; i < coords.length; i++) {
      ctx.lineTo(coords[i][0] * scaleX, coords[i][1] * scaleY)
    }

    ctx.strokeStyle = "red"
    ctx.lineWidth = 3
    ctx.stroke()

    parkingSlots.forEach((slot) => {

      ctx.beginPath()
      slot.points.forEach(([x, y], index) => {

        const scaledX = x * scaleX
        const scaledY = y * scaleY

        if (index === 0) {
          ctx.moveTo(scaledX, scaledY)
        } else {
          ctx.lineTo(scaledX, scaledY)
        }
      })

      ctx.closePath()
      ctx.fillStyle =
        slotStatus[slot.name] === "occupied"
          ? "rgba(255,0,0,0.45)"
          : "rgba(0,255,0,0.35)"
      ctx.fill()

      ctx.strokeStyle = "white"
      ctx.lineWidth = 2
      ctx.stroke()

      let centerX = 0
      let centerY = 0

      slot.points.forEach(([x, y]) => {
        centerX += x
        centerY += y
      })

      centerX = (centerX / slot.points.length) * scaleX
      centerY = (centerY / slot.points.length) * scaleY

      ctx.fillStyle = "white"
      ctx.font = "bold 8px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(slot.name, centerX, centerY)
    })

    coords.forEach(([x, y]) => {
      ctx.beginPath()
      ctx.arc(x * scaleX, y * scaleY, 5, 0, 2 * Math.PI)
      ctx.fillStyle = "black"
      ctx.fill()
    })
  }

  useEffect(() => {
    setTimeout(() => {
      drawPath(pathCoords)
    }, 100)
  }, [pathCoords])

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
          setSlotStatus(data.slot_status || {})
          setError('')
        }
      } catch (e) {
        setError('Backend not running')
      }
    }

    fetchStatus()
    fetchPath()
    const interval = setInterval(() => {
      fetchStatus()
      fetchPath()
    }, 2000)
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: 'videos/aitd_parking_lot.mp4'
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
      const res = await fetch('/api/stop_camera', { method: 'POST' })
      if (res.ok) {
        setIsLive(false)
        setLiveFeedUrl('')
        setFeedKey(0)
        setError('')
      } else {
        setError('Failed to stop')
      }
    } catch (e) {
      setError('Error stopping')
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1>Smart Parking Dashboard</h1>
          <p>Real-time parking monitoring and analytics</p>
        </div>
        <div className="live-badge">
          <div className="live-dot"></div>
          LIVE
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
        </div>
      )}

      <div className="tab-navigation">
        <button 
          className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`} 
          onClick={() => setActiveTab('live')}
        >
          Live Monitoring
        </button>
        <button 
          className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} 
          onClick={() => setActiveTab('history')}
        >
          Vehicle History Logs
        </button>
      </div>

      {activeTab === 'live' ? (
        <>
          <section className="stats">
            <div className="card">
              <div className="card-info">
                <label>Total Slots</label>
                <div className="val val-blue">{totalSlots}</div>
              </div>
            </div>
            <div className="card">
              <div className="card-info">
                <label>Occupied</label>
                <div className="val val-red">{occupiedSlots}</div>
              </div>
            </div>
            <div className="card">
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
                    Start
                  </button>
                ) : (
                  <button className="btn-stop" onClick={stopVideo}>
                    Stop
                  </button>
                )}
                <a
                  href="/api/live_feed"
                  target="_blank"
                  rel="noreferrer"
                  className="btn-open"
                >
                  Open Feed
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
                    <span>Occupied</span>
                    <span className="slot-num">Slot 1</span>
                  </div>
                  <div className="cam-slot occ">
                    <span>Occupied</span>
                    <span className="slot-num">Slot 2</span>
                  </div>
                  <div className="cam-slot occ">
                    <span>Occupied</span>
                    <span className="slot-num">Slot 3</span>
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <span className="panel-title">
                Smart Navigation
              </span>
            </div>
            <p className="best-slot-text">
              Best Slot:
              {bestSlot === "FULL"
                ? " Parking Full 🚫"
                : ` ${bestSlot}`}
            </p>
            <div className="parking-map-wrapper">
              <img
                src="./aitd_parking_lot_main.png"
                className="parking-map"
                crossOrigin="anonymous"
              />
              <canvas
                id="pathCanvas"
                width="900"
                height="440"
                className="path-canvas"
              ></canvas>
              {bestSlot === "FULL" && (
                <div className="full-overlay">
                  Parking Full 🚫
                </div>
              )}
            </div>
          </section>
        </>
      ) : (
        <div className="history-container">
          <div className="filters-bar">
            <div className="filter-group">
              <label>Filter by Date</label>
              <input 
                type="date" 
                className="filter-input" 
                value={filterDate} 
                onChange={(e) => setFilterDate(e.target.value)}
              />
            </div>
            <div className="filter-group">
              <label>Filter by Time</label>
              <input 
                type="time" 
                className="filter-input" 
                value={filterTime} 
                onChange={(e) => setFilterTime(e.target.value)}
              />
            </div>
            {(filterDate || filterTime) && (
              <button 
                className="btn-reset" 
                onClick={() => { setFilterDate(''); setFilterTime(''); }}
              >
                Reset Filters
              </button>
            )}
          </div>

          {sortedDates.length === 0 ? (
            <div className="no-data">
              No vehicle tracking history found matching the selected filters.
            </div>
          ) : (
            sortedDates.map(dateStr => (
              <div key={dateStr} className="date-group">
                <div className="date-header">
                  <span>{formatDate(dateStr)}</span>
                  <span className="date-count">{groupedLogs[dateStr].length} record(s)</span>
                </div>
                <div className="table-wrapper">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Track ID</th>
                        <th>Plate Number</th>
                        <th>Slot ID</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Current Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {groupedLogs[dateStr].map((log, idx) => (
                        <tr key={idx}>
                          <td><strong>#{log.track_id}</strong></td>
                          <td>
                            <span style={{ fontFamily: 'monospace', letterSpacing: '0.5px', background: 'rgba(255,255,255,0.05)', padding: '2px 6px', borderRadius: '4px' }}>
                              {log.plate_number || 'UNKNOWN'}
                            </span>
                          </td>
                          <td>{log.slot_id !== null ? `Slot ${log.slot_id}` : '-'}</td>
                          <td>{formatTime(log.entry_time)}</td>
                          <td>{formatTime(log.exit_time)}</td>
                          <td>
                            <span className={`badge-status status-${log.status}`}>
                              {log.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}

export default App

