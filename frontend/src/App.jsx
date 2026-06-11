import { useState, useEffect, useRef, useCallback } from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, CartesianGrid, LineChart, Line } from 'recharts'

const POLL_INTERVAL = 2000

export default function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [snapshot, setSnapshot] = useState({ slots: [], stats: {}, events: [], alerts: [] })
  const [sections, setSections] = useState([])
  const [selectedSectionId, setSelectedSectionId] = useState('ground_floor')
  const [analytics, setAnalytics] = useState({ status: {}, slot_utilization: [], peak_hours: [], avg_dwell_mins: 0 })
  const [logs, setLogs] = useState({ events: [] })
  const [alerts, setAlerts] = useState([])
  const [selectedSlot, setSelectedSlot] = useState(null)
  
  // Controls
  const [pipelineRunning, setPipelineRunning] = useState(true)
  const [connError, setConnError] = useState(false)
  const [now, setNow] = useState(new Date())
  
  // Search Filters (used for API queries)
  const [filterPlate, setFilterPlate] = useState('')
  const [filterSlot, setFilterSlot] = useState('')
  const [filterEventType, setFilterEventType] = useState('')

  // Local inputs state to prevent race conditions during typing
  const [inputPlate, setInputPlate] = useState('')
  const [inputSlot, setInputSlot] = useState('')
  const [inputType, setInputType] = useState('')

  // System clock
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  // Poll Snapshots
  const fetchSnapshot = useCallback(async () => {
    try {
      const res = await fetch('/api/snapshot')
      if (!res.ok) throw new Error('API Error')
      const data = await res.json()
      setSnapshot(data)
      setAlerts(data.alerts || [])
      setConnError(false)
    } catch (e) {
      setConnError(true)
    }
  }, [])

  // Fetch sections configuration
  const fetchSections = useCallback(async () => {
    try {
      const res = await fetch('/api/control/sections')
      if (res.ok) {
        const data = await res.json()
        setSections(data.sections || [])
      }
    } catch (e) {
      console.error(e)
    }
  }, [])

  useEffect(() => {
    fetchSnapshot()
    fetchSections()
    const timer = setInterval(() => {
      fetchSnapshot()
    }, POLL_INTERVAL)
    return () => clearInterval(timer)
  }, [fetchSnapshot, fetchSections])

  // Fetch Analytics on Tab Switch
  const fetchAnalytics = async () => {
    try {
      const res = await fetch('/api/analytics')
      if (res.ok) {
        const data = await res.json()
        setAnalytics(data)
      }
    } catch (e) {
      console.error('Failed to fetch analytics:', e)
    }
  }

  // Fetch Logs on Tab Switch or Filter Change
  const fetchLogs = useCallback(async (plate, slot, type) => {
    try {
      let url = '/api/events?limit=100'
      if (plate) url += `&plate=${encodeURIComponent(plate)}`
      if (slot) url += `&slot_id=${encodeURIComponent(slot)}`
      if (type) url += `&event_type=${encodeURIComponent(type)}`

      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        setLogs(data)
      }
    } catch (e) {
      console.error('Failed to fetch logs:', e)
    }
  }, [])

  useEffect(() => {
    if (activeTab === 'analytics') {
      fetchAnalytics()
    } else if (activeTab === 'logs') {
      fetchLogs(filterPlate, filterSlot, filterEventType)
    }
  }, [activeTab, fetchLogs, filterPlate, filterSlot, filterEventType])

  const switchSection = async (sectionId) => {
    try {
      const res = await fetch('/api/control/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ section_id: sectionId })
      })
      if (res.ok) {
        setSelectedSectionId(sectionId)
        setPipelineRunning(true)
        setTimeout(() => {
          fetchSnapshot()
        }, 1000)
      }
    } catch (e) {
      console.error(e)
    }
  }

  const startPipeline = async () => {
    await switchSection(selectedSectionId)
  }

  const stopPipeline = async () => {
    try {
      const res = await fetch('/api/control/stop', { method: 'POST' })
      if (res.ok) {
        setPipelineRunning(false)
      }
    } catch (e) {
      console.error(e)
    }
  }

  const calibrateSlots = async () => {
    try {
      const res = await fetch('/api/control/calibrate', { method: 'POST' })
      if (res.ok) {
        const data = await res.json()
        if (data.error) {
          alert(`Error launching calibrator: ${data.error}`)
          return
        }
        setPipelineRunning(false)
        alert('Calibration window opened on your desktop! Left-Click to draw points, ENTER to finish each slot, and S to save and exit.')
      }
    } catch (e) {
      console.error(e)
    }
  }

  // Logs search and clear handlers
  const handleSearch = () => {
    setFilterPlate(inputPlate)
    setFilterPlate(inputPlate)
    setFilterSlot(inputSlot)
    setFilterEventType(inputType)
    fetchLogs(inputPlate, inputSlot, inputType)
  }

  const handleReset = () => {
    setInputPlate('')
    setInputSlot('')
    setInputType('')
    setFilterPlate('')
    setFilterSlot('')
    setFilterEventType('')
    fetchLogs('', '', '')
  }

  const handleClearLogs = async () => {
    if (!window.confirm("Are you sure you want to clear all history logs and reset parking slots? This cannot be undone.")) {
      return
    }
    try {
      const res = await fetch('/api/events/clear', { method: 'POST' })
      if (res.ok) {
        const data = await res.json()
        if (data.status === 'success') {
          alert('All logs and slot states reset successfully!')
          setInputPlate('')
          setInputSlot('')
          setInputType('')
          setFilterPlate('')
          setFilterSlot('')
          setFilterEventType('')
          fetchLogs('', '', '')
          fetchSnapshot()
        } else {
          alert(`Failed to clear logs: ${data.message}`)
        }
      } else {
        alert('Failed to clear logs from database.')
      }
    } catch (e) {
      console.error(e)
    }
  }

  // Formatting Helpers
  const formatTime = (isoStr) => {
    if (!isoStr) return '-'
    const d = new Date(isoStr)
    return d.toLocaleTimeString('en-IN', { hour12: false })
  }

  const formatDwell = (secs) => {
    if (!secs) return '-'
    const m = Math.floor(secs / 60)
    const s = secs % 60
    return m > 0 ? `${m}m ${s}s` : `${s}s`
  }

  const { stats = {}, slots = [], events = [] } = snapshot
  const occupiedCount = stats.occupied || 0
  const freeCount = stats.free || 0
  const totalCount = stats.total || 0
  const occupancyRate = totalCount > 0 ? Math.round((occupiedCount / totalCount) * 100) : 0

  return (
    <div className="app-container">
      {/* ── Header Bar ── */}
      <header className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div className={`pulse-dot ${connError ? 'red' : 'green'}`} />
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 700 }}>Smart Parking Control Hub</h1>
            <p className="text-muted" style={{ fontSize: 11, marginTop: 2 }}>NextGen Computer Vision & Navigation</p>
          </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, fontSize: 12 }}>
          <div style={{ textAlign: 'right' }}>
            <div className="text-muted" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Current Feed</div>
            <div style={{ fontWeight: 600, color: '#60a5fa', marginTop: 2 }}>{stats.current_source || 'No Stream'}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="text-muted" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Performance</div>
            <div style={{ fontWeight: 600, color: '#34d399', marginTop: 2 }}>{stats.fps != null ? `${stats.fps} FPS` : '0 FPS'}</div>
          </div>
          <div style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)', borderLeft: '1px solid var(--border)', paddingLeft: 24 }}>
            <div>{now.toLocaleDateString('en-IN')}</div>
            <div style={{ fontSize: 13, color: '#f8fafc', fontWeight: 500, marginTop: 2 }}>{now.toLocaleTimeString('en-IN', { hour12: false })}</div>
          </div>
        </div>
      </header>

      {/* ── Navigation Tabs ── */}
      <div style={{ display: 'flex', gap: 12 }}>
        {[
          { id: 'live', name: 'Live Stream & Smart Suggestion' },
          { id: 'slots', name: 'Slots Overview' },
          { id: 'alerts', name: 'Parking Alerts' },
          { id: 'sections', name: 'Sections Configurator' },
          { id: 'logs', name: 'History Logs' }
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className="btn"
            style={{
              background: activeTab === t.id ? 'var(--primary-bg)' : 'rgba(255,255,255,0.02)',
              borderColor: activeTab === t.id ? 'var(--primary)' : 'var(--border)',
              color: activeTab === t.id ? '#60a5fa' : 'var(--text)',
              padding: '12px 20px',
              borderRadius: 'var(--radius-sm)'
            }}
          >
            {t.name}
          </button>
        ))}
      </div>

      {/* ── Stats Row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {[
          { label: 'Available Slots', value: freeCount, color: '#10b981', subText: `${totalCount - occupiedCount} spots free` },
          { label: 'Occupied Spots', value: occupiedCount, color: '#ef4444', subText: `${occupancyRate}% utilization` },
          { label: 'Total Capacity', value: totalCount, color: '#3b82f6', subText: 'Configured parking slots' }
        ].map((s, idx) => (
          <div key={idx} className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span className="text-muted" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{s.label}</span>
            <span style={{ fontSize: 32, fontWeight: 700, color: s.color }}>{s.value}</span>
            <span className="text-muted" style={{ fontSize: 11 }}>{s.subText}</span>
          </div>
        ))}
      </div>

      {/* ── Main Tab Contents ── */}
      
      {/* 1. Live Tab */}
      {activeTab === 'live' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20 }}>
          {/* Left: Stream View & Pipeline Controls */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyItems: 'center', justifyContent: 'space-between' }}>
              <h2 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="pulse-dot green" /> Video Monitor
              </h2>
              {/* Dynamic Pipeline Controls */}
              <div style={{ display: 'flex', gap: 12 }}>
                <select
                  value={selectedSectionId}
                  onChange={(e) => switchSection(e.target.value)}
                  className="input"
                  style={{ background: '#131924', border: '1px solid var(--border)', minWidth: '150px' }}
                >
                  {sections.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
                
                <button className="btn btn-secondary" onClick={calibrateSlots}>Calibrate Spots</button>

                {pipelineRunning ? (
                  <button className="btn btn-danger" onClick={stopPipeline}>Stop Pipeline</button>
                ) : (
                  <button className="btn btn-primary" onClick={startPipeline}>Start Feed</button>
                )}
              </div>
            </div>

            <div style={{ background: '#020408', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', overflow: 'hidden', aspectRatio: '16/9', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {pipelineRunning ? (
                <img
                  src={`/video_feed?t=${Date.now()}`}
                  alt="Live Feed Stream"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  onError={(e) => {
                    e.target.src = 'https://images.unsplash.com/photo-1506521781263-d8422e82f27a?auto=format&fit=crop&w=800&q=80'
                  }}
                />
              ) : (
                <div className="text-muted" style={{ textAlign: 'center' }}>
                  <svg width="48" height="48" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" style={{ margin: '0 auto 12px', opacity: 0.4 }}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25v13.5m-7.5-13.5v13.5" />
                  </svg>
                  <span>Video pipeline stopped. Start feed to resume monitoring.</span>
                </div>
              )}
            </div>
          </div>

          {/* Right: Smart Suggestion Assistant Card */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: 12 }}>
              <h2 style={{ fontSize: 20, fontWeight: 700, color: '#60a5fa' }}>Smart Parking Assistant</h2>
              <p className="text-muted" style={{ fontSize: 12, marginTop: 4 }}>Real-time section suggestions and active routing alerts.</p>
            </div>

            {/* Recommendation Display */}
            {(() => {
              const activeSec = sections.find(s => s.id === selectedSectionId);
              if (!activeSec) return <div className="text-muted">Loading recommendations...</div>;

              // Find occupied status for all slots
              const freeSlotsInActive = (activeSec.slots || [])
                .filter(s => {
                  const state = snapshot.slots.find(snap => String(snap.slot_id) === String(s.id));
                  return state ? state.status === 'free' : true;
                })
                .sort((a, b) => a.distance - b.distance);

              if (freeSlotsInActive.length > 0) {
                const best = freeSlotsInActive[0];
                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1.5px solid #10b981', padding: 18, borderRadius: 'var(--radius-sm)', boxShadow: '0 4px 20px rgba(16, 185, 129, 0.15)' }}>
                      <div className="text-muted" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Recommended Spot</div>
                      <div style={{ fontSize: 28, fontWeight: 800, color: '#10b981', margin: '6px 0' }}>Slot {best.id}</div>
                      <div style={{ fontSize: 13, color: '#f8fafc' }}>
                        Located on <strong>{activeSec.name}</strong>.
                      </div>
                    </div>
                    <p className="text-muted" style={{ fontSize: 12 }}>Slots sorted by proximity rank in configurations. Drive directly to the highlighted spot.</p>
                  </div>
                );
              }

              // Active floor is full, search other floors
              let redirectionSpot = null;
              let redirectionSec = null;

              for (const sec of sections) {
                if (sec.id === selectedSectionId) continue;
                const freeSlots = (sec.slots || [])
                  .filter(s => {
                    const state = snapshot.slots.find(snap => String(snap.slot_id) === String(s.id));
                    return state ? state.status === 'free' : true;
                  })
                  .sort((a, b) => a.distance - b.distance);

                if (freeSlots.length > 0) {
                  if (!redirectionSpot || freeSlots[0].distance < redirectionSpot.distance) {
                    redirectionSpot = freeSlots[0];
                    redirectionSec = sec;
                  }
                }
              }

              if (redirectionSpot && redirectionSec) {
                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1.5px solid #ef4444', padding: 18, borderRadius: 'var(--radius-sm)', boxShadow: '0 4px 20px rgba(239, 68, 68, 0.15)' }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#ef4444', textTransform: 'uppercase' }}>Section Full Redirect</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc', margin: '8px 0' }}>{activeSec.name} is completely occupied!</div>
                      <div style={{ fontSize: 13, color: '#f8fafc', borderTop: '1px solid rgba(239,68,68,0.2)', paddingTop: 10, marginTop: 4 }}>
                        Redirecting to nearest spot: <strong style={{ color: '#3b82f6' }}>Slot {redirectionSpot.id}</strong> on <strong>{redirectionSec.name}</strong>.
                      </div>
                    </div>
                    <button className="btn btn-primary" onClick={() => switchSection(redirectionSec.id)}>Switch Feed to {redirectionSec.name}</button>
                  </div>
                );
              }

              return (
                <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1.5px solid #ef4444', padding: 20, borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#ef4444' }}>Parking Lot Full</div>
                  <div className="text-muted" style={{ fontSize: 12, marginTop: 4 }}>No empty spaces found across all sections/floors.</div>
                </div>
              );
            })()}

            {/* Quick Section Switcher Info */}
            <div className="glass-panel" style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border)', padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600 }}>Active Feeds Monitor</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {sections.map(s => {
                  const isActive = s.id === selectedSectionId;
                  
                  // Compute occupancy count
                  const totalSlots = s.slots ? s.slots.length : 0;
                  const occupiedCount = (s.slots || []).filter(sl => {
                    const state = snapshot.slots.find(snap => String(snap.slot_id) === String(sl.id));
                    return state ? state.status === 'occupied' : false;
                  }).length;
                  const isFull = occupiedCount === totalSlots;

                  return (
                    <div
                      key={s.id}
                      onClick={() => switchSection(s.id)}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '10px 14px',
                        background: isActive ? 'rgba(59, 130, 246, 0.1)' : 'rgba(255,255,255,0.02)',
                        border: `1px solid ${isActive ? '#3b82f6' : 'transparent'}`,
                        borderRadius: 6,
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => { if(!isActive) e.currentTarget.style.background = 'rgba(255,255,255,0.05)' }}
                      onMouseLeave={(e) => { if(!isActive) e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <span style={{ fontSize: 12, fontWeight: 600, color: isActive ? '#60a5fa' : '#f8fafc' }}>{s.name}</span>
                        <span className="text-muted" style={{ fontSize: 9, fontFamily: 'monospace' }}>{s.source}</span>
                      </div>
                      <span style={{ fontSize: 11, fontWeight: 600, color: isFull ? '#ef4444' : '#10b981' }}>
                        {occupiedCount}/{totalSlots} spots
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2. Slots Overview Tab */}
      {activeTab === 'slots' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 20 }}>
          {/* Left: Slot Grid */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <h2>Configured Slots Status</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 14 }}>
              {slots.map(s => {
                const isOccupied = s.status === 'occupied'
                return (
                  <div
                    key={s.slot_id}
                    onClick={() => setSelectedSlot(s)}
                    style={{
                      background: isOccupied ? 'var(--red-bg)' : 'var(--green-bg)',
                      border: `1.5px solid ${isOccupied ? 'var(--red)' : 'var(--green)'}`,
                      borderRadius: 'var(--radius-sm)',
                      padding: 16,
                      textAlign: 'center',
                      cursor: 'pointer',
                      transition: 'transform 0.2s',
                      boxShadow: isOccupied ? '0 4px 12px var(--red-glow)' : '0 4px 12px var(--green-glow)'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.04)'}
                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                  >
                    <div style={{ fontSize: 16, fontWeight: 700, color: isOccupied ? 'var(--red)' : 'var(--green)' }}>Slot {s.slot_id}</div>
                    <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '6px 0' }}>{s.status}</div>
                    {isOccupied && (
                      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, background: 'rgba(0,0,0,0.2)', padding: '2px 4px', borderRadius: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {s.plate || `Track #${s.track_id}`}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Right: Slot Detailed Inspector */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <h2>Slot Detailed Inspector</h2>
            {selectedSlot ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: 12 }}>
                  <span style={{ fontSize: 20, fontWeight: 700 }}>Slot ID: {selectedSlot.slot_id}</span>
                  <span style={{ color: selectedSlot.status === 'occupied' ? 'var(--red)' : 'var(--green)', fontWeight: 600 }}>
                    {selectedSlot.status.toUpperCase()}
                  </span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="text-muted">Associated Track</span>
                    <span>{selectedSlot.track_id != null ? `#${selectedSlot.track_id}` : '-'}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="text-muted">Detected Plate</span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', background: 'rgba(255,255,255,0.05)', padding: '2px 6px', borderRadius: 4 }}>
                      {selectedSlot.plate || 'UNKNOWN'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-muted" style={{ textAlign: 'center', padding: '40px 0' }}>
                Select a slot from the overview grid to inspect occupant profiles.
              </div>
            )}
          </div>
        </div>
      )}

      {/* 3. Parking Alerts Tab */}
      {activeTab === 'alerts' && (
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2>Parking Violation Alerts</h2>
              <p className="text-muted" style={{ fontSize: 12, marginTop: 4 }}>Real-time detection of wrongly parked vehicles</p>
            </div>
            <button
              className="btn btn-danger"
              onClick={async () => {
                try {
                  const res = await fetch('/api/alerts/clear', { method: 'POST' })
                  if (res.ok) {
                    setAlerts([])
                  }
                } catch (e) {
                  console.error(e)
                }
              }}
            >
              Clear All Alerts
            </button>
          </div>

          {alerts.length === 0 ? (
            <div className="text-muted" style={{ textAlign: 'center', padding: '60px 0' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>🚗</div>
              <div style={{ fontSize: 16 }}>No active parking violations detected</div>
              <div style={{ fontSize: 12, marginTop: 8 }}>Alerts will appear here when vehicles are parked in non-designated areas</div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {alerts.map((alert, idx) => (
                <div
                  key={idx}
                  style={{
                    background: 'rgba(239, 68, 68, 0.08)',
                    border: '1.5px solid #ef4444',
                    borderRadius: 'var(--radius-sm)',
                    padding: 16,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    boxShadow: '0 4px 12px rgba(239, 68, 68, 0.15)'
                  }}
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <span style={{
                        padding: '4px 12px',
                        borderRadius: 12,
                        fontSize: 10,
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        background: '#ef4444',
                        color: '#fff'
                      }}>
                        {alert.type === 'wrong_parking' ? 'Wrong Parking' : alert.type}
                      </span>
                      <span style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc' }}>
                        Track #{alert.track_id}
                      </span>
                      {alert.plate && alert.plate !== 'Unknown' && (
                        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, background: 'rgba(255,255,255,0.1)', padding: '2px 8px', borderRadius: 4 }}>
                          {alert.plate}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                      {alert.message}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
                      Duration: {alert.duration_seconds}s • {formatTime(alert.timestamp)}
                    </div>
                  </div>
                  <div style={{
                    width: 48,
                    height: 48,
                    borderRadius: '50%',
                    background: 'rgba(239, 68, 68, 0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 24
                  }}>
                    ⚠️
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}



      {/* 4. History Logs Tab */}
      {activeTab === 'logs' && (
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>Vehicle Entrance & Exit Logs</h2>
            {/* Search Filters Bar */}
            <div style={{ display: 'flex', gap: 12 }}>
              <input
                type="text"
                placeholder="Search plate..."
                value={inputPlate}
                onChange={(e) => setInputPlate(e.target.value)}
                className="input"
              />
              <input
                type="text"
                placeholder="Slot ID..."
                value={inputSlot}
                onChange={(e) => setInputSlot(e.target.value)}
                className="input"
                style={{ width: 100 }}
              />
              <select
                value={inputType}
                onChange={(e) => setInputType(e.target.value)}
                className="input"
              >
                <option value="">All Events</option>
                <option value="entry">Entries</option>
                <option value="exiting">Exits</option>
                <option value="ocr_update">OCR Reads</option>
              </select>
              
              <button className="btn btn-primary" onClick={handleSearch}>Search</button>
              <button className="btn btn-secondary" onClick={handleReset}>Reset</button>
              <button className="btn btn-danger" onClick={handleClearLogs} style={{ marginLeft: 8 }}>Clear History</button>
            </div>
          </div>

          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Event ID</th>
                  <th>Timestamp</th>
                  <th>Track ID</th>
                  <th>Event Type</th>
                  <th>Slot ID</th>
                  <th>Plate Number</th>
                  <th>Confidence</th>
                  <th>Dwell Time</th>
                </tr>
              </thead>
              <tbody>
                {logs.events && logs.events.length > 0 ? (
                  logs.events.map((log) => {
                    const isEntry = log.event_type === 'entry'
                    const isExit = log.event_type === 'exiting'
                    return (
                      <tr key={log.id}>
                        <td><strong>#{log.id}</strong></td>
                        <td style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)' }}>{formatTime(log.timestamp)}</td>
                        <td><strong>Track {log.track_id}</strong></td>
                        <td>
                          <span style={{
                            padding: '2px 8px',
                            borderRadius: 12,
                            fontSize: 10,
                            fontWeight: 600,
                            textTransform: 'uppercase',
                            background: isEntry ? 'var(--green-bg)' : (isExit ? 'var(--red-bg)' : 'var(--primary-bg)'),
                            color: isEntry ? 'var(--green)' : (isExit ? 'var(--red)' : 'var(--primary)')
                          }}>
                            {log.event_type === 'ocr_update' ? 'OCR Read' : (isEntry ? 'Entry' : 'Exit')}
                          </span>
                        </td>
                        <td><strong>Slot {log.slot_id}</strong></td>
                        <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>
                          {log.plate ? (
                            <span style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border)', padding: '2px 6px', borderRadius: 4 }}>
                              {log.plate}
                            </span>
                          ) : '-'}
                        </td>
                        <td>
                          {log.ocr_conf != null ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <div style={{ width: 40, height: 3, background: 'rgba(255,255,255,0.1)', borderRadius: 2, overflow: 'hidden' }}>
                                <div style={{ width: `${Math.round(log.ocr_conf * 100)}%`, height: '100%', background: 'var(--green)' }} />
                              </div>
                              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{Math.round(log.ocr_conf * 100)}%</span>
                            </div>
                          ) : '-'}
                        </td>
                        <td style={{ color: 'var(--text-muted)' }}>{formatDwell(log.dwell_secs)}</td>
                      </tr>
                    )
                  })
                ) : (
                  <tr>
                    <td colSpan="8" style={{ textAlign: 'center', padding: 24 }} className="text-muted">
                      No records found matching filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 5. Sections Configurator Tab */}
      {activeTab === 'sections' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 20 }}>
          {/* Left: Sections List */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2>Configure Floors</h2>
              <button
                className="btn btn-primary"
                onClick={() => {
                  const newId = `floor_${Date.now()}`;
                  const newSec = {
                    id: newId,
                    name: 'New Section Floor',
                    source: '0',
                    slots_file: `backend/marked_slots/slots_${newId}.json`,
                    slots: [{ id: '9', distance: 10 }]
                  };
                  setSections([...sections, newSec]);
                  setSelectedSectionId(newId);
                }}
              >
                + Add Floor
              </button>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {sections.map(s => {
                const isActive = s.id === selectedSectionId;
                return (
                  <div
                    key={s.id}
                    onClick={() => setSelectedSectionId(s.id)}
                    style={{
                      padding: 14,
                      background: isActive ? 'rgba(59, 130, 246, 0.1)' : 'rgba(255,255,255,0.02)',
                      border: `1px solid ${isActive ? '#3b82f6' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-sm)',
                      cursor: 'pointer',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 600 }}>{s.name}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>ID: {s.id}</div>
                    </div>
                    <button
                      className="btn btn-danger"
                      style={{ padding: '4px 8px', fontSize: 10 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Delete section ${s.name}?`)) {
                          const updated = sections.filter(sec => sec.id !== s.id);
                          setSections(updated);
                          if (selectedSectionId === s.id && updated.length > 0) {
                            setSelectedSectionId(updated[0].id);
                          }
                        }
                      }}
                    >
                      Delete
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right: Section Detail Editor */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {(() => {
              const active = sections.find(s => s.id === selectedSectionId);
              if (!active) {
                return <div className="text-muted" style={{ textAlign: 'center', padding: '40px 0' }}>Select or add a floor section to edit parameters.</div>;
              }

              // Handlers to update local state field
              const updateField = (field, val) => {
                const updated = sections.map(s => {
                  if (s.id === active.id) {
                    return { ...s, [field]: val };
                  }
                  return s;
                });
                setSections(updated);
              };

              // Slots list edit helpers
              const addSlotRow = () => {
                const slots = [...(active.slots || []), { id: '', distance: 10 }];
                updateField('slots', slots);
              };

              const removeSlotRow = (index) => {
                const slots = (active.slots || []).filter((_, idx) => idx !== index);
                updateField('slots', slots);
              };

              const updateSlotValue = (index, key, val) => {
                const slots = (active.slots || []).map((s, idx) => {
                  if (idx === index) {
                    return { ...s, [key]: val };
                  }
                  return s;
                });
                updateField('slots', slots);
              };

              // Save sections settings to database file
              const handleSaveConfig = async () => {
                try {
                  const res = await fetch('/api/control/sections', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sections })
                  });
                  if (res.ok) {
                    alert('Floor configuration saved successfully!');
                    fetchSections();
                  } else {
                    alert('Failed to save floor configurations.');
                  }
                } catch (e) {
                  console.error(e);
                }
              };

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <h2>Edit Floor Section Details</h2>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <span className="text-muted" style={{ fontSize: 11 }}>Floor Section Name</span>
                      <input
                        type="text"
                        value={active.name}
                        onChange={(e) => updateField('name', e.target.value)}
                        className="input"
                      />
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <span className="text-muted" style={{ fontSize: 11 }}>Camera Source (Video File or Webcam Index)</span>
                      <input
                        type="text"
                        value={active.source}
                        onChange={(e) => updateField('source', e.target.value)}
                        className="input"
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <span className="text-muted" style={{ fontSize: 11 }}>Floor Section ID (Alpha-numeric, unique)</span>
                      <input
                        type="text"
                        value={active.id}
                        disabled
                        className="input"
                        style={{ opacity: 0.5, cursor: 'not-allowed' }}
                      />
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <span className="text-muted" style={{ fontSize: 11 }}>Slot Calibration JSON File Path</span>
                      <input
                        type="text"
                        value={active.slots_file}
                        onChange={(e) => updateField('slots_file', e.target.value)}
                        className="input"
                      />
                    </div>
                  </div>

                  {/* Slot Proximity Configuration */}
                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                      <h3>Slot Priority & Distance Configuration</h3>
                      <button className="btn btn-secondary" onClick={addSlotRow} style={{ padding: '6px 12px', fontSize: 11 }}>+ Add Slot Mapping</button>
                    </div>

                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Slot ID</th>
                          <th>Distance from Entrance (meters)</th>
                          <th style={{ width: 60 }}>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(active.slots || []).map((s, idx) => (
                          <tr key={idx}>
                            <td>
                              <input
                                type="text"
                                value={s.id}
                                onChange={(e) => updateSlotValue(idx, 'id', e.target.value)}
                                className="input"
                                style={{ width: '100px', height: '30px', fontSize: 12 }}
                                placeholder="e.g. 1"
                              />
                            </td>
                            <td>
                              <input
                                type="number"
                                value={s.distance}
                                onChange={(e) => updateSlotValue(idx, 'distance', parseInt(e.target.value) || 0)}
                                className="input"
                                style={{ width: '100px', height: '30px', fontSize: 12 }}
                                placeholder="e.g. 10"
                              />
                            </td>
                            <td>
                              <button className="btn btn-danger" onClick={() => removeSlotRow(idx)} style={{ padding: '4px 8px', fontSize: 10 }}>Remove</button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Action buttons */}
                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
                    <button className="btn btn-secondary" onClick={calibrateSlots}>Calibrate Slots Polygons</button>
                    <button className="btn btn-primary" onClick={handleSaveConfig}>Save All Configuration</button>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  )
}
