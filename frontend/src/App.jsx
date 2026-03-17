import { useState, useEffect } from 'react'
import './index.css'

function App() {
  const [totalSlots, setTotalSlots] = useState(0)
  const [occupiedSlots, setOccupiedSlots] = useState(0)
  const [freeSlots, setFreeSlots] = useState(0)
  const [capacityUsage, setCapacityUsage] = useState(0)
  const [showParked, setShowParked] = useState(false)
  const [cameraUrl, setCameraUrl] = useState('http://localhost:5000/video')
  const [resultImageUrl, setResultImageUrl] = useState('http://localhost:5000/latest_result')
  const [resultImages, setResultImages] = useState([])

  // Fetch parking data from backend
  useEffect(() => {
    const fetchParkingData = async () => {
      try {
        const response = await fetch('http://localhost:5000/status')
        if (response.ok) {
          const data = await response.json()
          setTotalSlots(data.total_slots || 0)
          setOccupiedSlots(data.occupied || 0)
          setFreeSlots(data.free || 0)
          setCapacityUsage(data.occupancy_rate || 0)
        }
      } catch (error) {
        console.error('Error fetching parking data:', error)
        // Set default values if backend not available
        setTotalSlots(6)
        setOccupiedSlots(2)
        setFreeSlots(4)
        setCapacityUsage(33)
      }
    }

    // Fetch result images
    const fetchResultImages = async () => {
      try {
        const response = await fetch('http://localhost:5000/results')
        if (response.ok) {
          const data = await response.json()
          setResultImages(data.results || [])
        }
      } catch (error) {
        console.error('Error fetching result images:', error)
      }
    }

    fetchParkingData()
    fetchResultImages()
    const interval = setInterval(() => {
      fetchParkingData()
      fetchResultImages()
    }, 2000) // Update every 2 seconds

    return () => clearInterval(interval)
  }, [])

  const parkedVehicles = [
    { id: 1, vehicleNo: 'MH12AB1234', slot: 'A1', entryTime: '10:15 AM', status: 'Parked' },
    { id: 2, vehicleNo: 'MH14CD5678', slot: 'B3', entryTime: '11:05 AM', status: 'Parked' },
    { id: 3, vehicleNo: 'MH01EF9012', slot: 'C2', entryTime: '12:30 PM', status: 'Parked' },
  ]

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Parking Dashboard</h1>
          <p>Live overview of parking slots and camera feed</p>
        </div>
        <div className="header-actions">
          <button
            className={showParked ? 'header-button' : 'header-button header-button-active'}
            onClick={() => setShowParked(false)}
          >
            Dashboard
          </button>
          <button
            className={showParked ? 'header-button header-button-active' : 'header-button'}
            onClick={() => setShowParked(true)}
          >
            Parked Vehicles
          </button>
        </div>
      </header>

      <main className="app-main">
        {showParked ? (
          <section className="content-grid">
            <div className="panel panel-parked">
              <h2>Parked Vehicles</h2>
              <p className="capacity-summary">Currently parked vehicles in the lot.</p>
              <div className="table-wrapper">
                <table className="parked-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Vehicle No</th>
                      <th>Slot</th>
                      <th>Entry Time</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {parkedVehicles.map((vehicle, index) => (
                      <tr key={vehicle.id}>
                        <td>{index + 1}</td>
                        <td>{vehicle.vehicleNo}</td>
                        <td>{vehicle.slot}</td>
                        <td>{vehicle.entryTime}</td>
                        <td>{vehicle.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        ) : (
          <>
            <section className="stats">
              <div className="stat-card stat-total">
                <h2>Total Slots</h2>
                <p className="stat-value">{totalSlots}</p>
              </div>
              <div className="stat-card stat-occupied">
                <h2>Occupied</h2>
                <p className="stat-value">{occupiedSlots}</p>
              </div>
              <div className="stat-card stat-free">
                <h2>Free</h2>
                <p className="stat-value">{freeSlots}</p>
              </div>
            </section>

            <section className="capacity-section">
              <div className="panel capacity-panel">
                <h2>Total Capacity</h2>
                <p className="capacity-summary">
                  {occupiedSlots} of {totalSlots} slots occupied ({capacityUsage}%)
                </p>
                <div className="capacity-bar">
                  <div
                    className="capacity-bar-fill"
                    style={{ width: `${capacityUsage}%` }}
                  ></div>
                </div>
              </div>
            </section>

            <section className="content-grid">
              <div className="panel panel-camera">
                <h2>Live Camera</h2>
                <div className="camera-frame">
                  <img 
                    src={cameraUrl} 
                    alt="Live Parking Camera" 
                    style={{ width: '100%', height: 'auto', borderRadius: '8px' }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'block';
                    }}
                  />
                  <div style={{ display: 'none', textAlign: 'center', padding: '20px' }}>
                    <p>🚗 Live Parking Detection</p>
                    <p>Camera feed loading...</p>
                    <p style={{ fontSize: '12px', color: '#666' }}>
                      Make sure backend is running on localhost:5000
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="panel panel-results">
                <h2>Detection Results</h2>
                <div className="results-frame">
                  <img 
                    src={resultImageUrl} 
                    alt="Latest Detection Result" 
                    style={{ width: '100%', height: 'auto', borderRadius: '8px' }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'block';
                    }}
                  />
                  <div style={{ display: 'none', textAlign: 'center', padding: '20px' }}>
                    <p>📸 Detection Results</p>
                    <p>Processing detection results...</p>
                    <p style={{ fontSize: '12px', color: '#666' }}>
                      Results update every 2 seconds
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section className="content-grid">
              <div className="panel panel-history">
                <h2>Recent Results</h2>
                <div className="history-grid">
                  {resultImages.slice(0, 6).map((image, index) => (
                    <div key={index} className="history-item">
                      <img 
                        src={`http://localhost:5000/result/${image}`}
                        alt={`Result ${image}`}
                        style={{ width: '100%', height: '120px', objectFit: 'cover', borderRadius: '4px' }}
                      />
                      <p style={{ fontSize: '10px', margin: '4px 0' }}>{image}</p>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  )
}

export default App
