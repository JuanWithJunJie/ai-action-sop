import { useState, useEffect, useRef } from 'react'
import { io } from 'socket.io-client'

const STEPS = [
  { id: 'D1', name: '取料', en: 'D1_pick_material' },
  { id: 'D2', name: '撕膜', en: 'D2_tear_film' },
  { id: 'D3', name: '检测', en: 'D3_inspect' },
  { id: 'D4', name: '放料', en: 'D4_place_material' },
]

const STEP_STATUS = { pending: '未开始', active: '进行中', done: '已完成', timeout: '超时跳过' }

function StepCard({ step, index, status, confidence }) {
  const cls = status === 'active' ? 'active' : status === 'done' ? 'done' : status === 'timeout' ? 'timeout' : ''
  const statusColor = { active: 'amber', done: 'green', timeout: 'red' }[status]
  return (
    <div className={`step-card ${cls}`}>
      <div className="step-header">
        <div className="step-num">{step.id}</div>
        <div>
          <div className="step-name">{step.name}</div>
          <div className="step-en">{step.en}</div>
        </div>
      </div>
      <div className="step-status">
        <div className="status-dot"></div>
        <span style={{ color: statusColor ? `var(--${statusColor})` : 'var(--text-muted)' }}>
          {STEP_STATUS[status] || '未开始'}
        </span>
      </div>
      {status === 'active' && (
        <div className="step-conf">
          置信度: {(confidence * 100).toFixed(1)}%
          <div className="confidence-bar" style={{ marginTop: 4 }}>
            <div className="confidence-fill" style={{ width: `${confidence * 100}%` }}></div>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricBox({ label, value, color }) {
  return (
    <div className="metric-box">
      <div className="lbl">{label}</div>
      <div className={`val ${color || ''}`}>{value}</div>
    </div>
  )
}

function PassRateRing({ rate }) {
  const c = 2 * Math.PI * 25
  const offset = c - (rate / 100) * c
  const color = rate >= 90 ? '#00ff88' : rate >= 70 ? '#ffaa00' : '#ff4466'
  return (
    <div className="pass-rate-ring">
      <svg width="60" height="60">
        <circle cx="30" cy="30" r="25" stroke="rgba(0,212,255,0.1)" strokeWidth="4" fill="none" />
        <circle cx="30" cy="30" r="25" stroke={color} strokeWidth="4" fill="none" strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round" />
      </svg>
      <div className="ring-val" style={{ position: 'absolute' }}>{rate}%</div>
    </div>
  )
}

export default function App() {
  const [clock, setClock] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [frameData, setFrameData] = useState(null)
  const [status, setStatus] = useState({
    current_pred: '-', expected_show: '-', expected_conf: 0,
    hit: 0, confirm_frames: 4, top3: '-', frame_id: 0, time_sec: 0,
    progress: 0, total: 4, cycle: 1, events_count: 0,
  })
  const [stepStatuses, setStepStatuses] = useState(['pending', 'pending', 'pending', 'pending'])
  const [events, setEvents] = useState([])
  const [videoName, setVideoName] = useState('')
  const [passCount, setPassCount] = useState(0)
  const [skipCount, setSkipCount] = useState(0)
  const [yoloConf, setYoloConf] = useState('0.30')
  const [lstmConf, setLstmConf] = useState('0.15')
  const socketRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    const timer = setInterval(() => {
      setClock(new Date().toTimeString().slice(0, 8))
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    const socket = io('http://localhost:5000', { transports: ['websocket'] })
    socketRef.current = socket

    socket.on('init', (data) => {
      setStepStatuses(prev => data.total === 4 ? ['active', ...prev.slice(1).map(() => 'pending')] : prev)
    })

    socket.on('frame_status', (data) => {
      setStatus(data)
      const p = data.progress
      setStepStatuses(prev => {
        const next = [...prev]
        for (let i = 0; i < next.length; i++) {
          if (i < p) next[i] = prev[i] === 'timeout' ? 'timeout' : 'done'
          else if (i === p) next[i] = 'active'
          else next[i] = 'pending'
        }
        return next
      })
    })

    socket.on('frame_data', (data) => {
      const blob = new Blob([data], { type: 'image/jpeg' })
      setFrameData(URL.createObjectURL(blob))
    })

    socket.on('action', (data) => {
      if (data.status === '完成') setPassCount(c => c + 1)
      else setSkipCount(c => c + 1)
      if (data.event) {
        setEvents(prev => [{
          type: data.status === '完成' ? 'done' : 'skip',
          step: STEPS[data.index],
          cycle: data.event.cycle,
          time: data.event.done_time_sec,
          timestamp: data.event.done_time_beijing?.slice(11) || '',
        }, ...prev].slice(0, 30))
      }
      if (data.status === '超时跳过') {
        setStepStatuses(prev => {
          const next = [...prev]
          if (data.index < next.length) next[data.index] = 'timeout'
          if (data.index + 1 < next.length) next[data.index + 1] = 'active'
          return next
        })
      }
    })

    socket.on('new_cycle', () => {
      setStepStatuses(['active', 'pending', 'pending', 'pending'])
    })

    socket.on('finished', () => {
      setIsRunning(false)
    })

    socket.on('error', (data) => {
      alert(data.msg)
      setIsRunning(false)
    })

    return () => socket.disconnect()
  }, [])

  const handleUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const formData = new FormData()
    formData.append('video', file)
    fetch('http://localhost:5000/api/upload', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        setVideoName(data.name)
        setPassCount(0)
        setSkipCount(0)
        setEvents([])
        setStepStatuses(['pending', 'pending', 'pending', 'pending'])
      })
      .catch(err => alert('上传失败: ' + err))
  }

  const handleStart = () => {
    if (!videoName) {
      alert('请先导入视频')
      fileInputRef.current?.click()
      return
    }
    if (isRunning) {
      socketRef.current?.emit('stop_inference')
      setIsRunning(false)
    } else {
      fetch('http://localhost:5000/api/params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yolo_conf: parseFloat(yoloConf), lstm_conf: parseFloat(lstmConf) }),
      }).then(() => {
        setStepStatuses(['active', 'pending', 'pending', 'pending'])
        setPassCount(0)
        setSkipCount(0)
        setEvents([])
        setIsRunning(true)
        socketRef.current?.emit('start_inference', { path: 'D:\\ai-work\\AI-SOP-main\\AI-SOP-main\\uploads\\' + videoName })
      })
    }
  }

  const passRate = (passCount + skipCount) > 0 ? Math.round((passCount / (passCount + skipCount)) * 100) : 100

  return (
    <div className="app grid-bg">
      <div className="header">
        <div className="header-left">
          <div className="logo">
            <div className="logo-icon">AI</div>
            <div className="logo-text">SOP<span>·VISION</span></div>
          </div>
          <div className="divider-v"></div>
          <div className="header-info">
            <div className="info-item"><div className="info-label">工厂</div><div className="info-value">深圳智造工厂</div></div>
            <div className="info-item"><div className="info-label">产线</div><div className="info-value">A线 · 手机组装</div></div>
            <div className="info-item"><div className="info-label">工位</div><div className="info-value">W-07 镜面贴合</div></div>
            <div className="info-item"><div className="info-label">班次</div><div className="info-value">白班 A组</div></div>
          </div>
        </div>
        <div className="header-right">
          <div className="status-pill status-online"><div className="dot"></div><span>{isRunning ? '推理中' : '系统在线'}</span></div>
          <div className="clock mono">{clock}</div>
        </div>
      </div>

      <div className="main">
        <div className="left-col">
          <div className="video-section">
            <div className="corner-deco tl"></div><div className="corner-deco tr"></div>
            <div className="corner-deco bl"></div><div className="corner-deco br"></div>
            {isRunning && <div className="scan-line"></div>}
            {frameData ? (
              <img src={frameData} alt="frame" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            ) : (
              <div className="video-placeholder">
                <div className="icon">{videoName ? '✓' : '📷'}</div>
                <div className="text">{videoName ? `已加载: ${videoName}` : '点击「导入视频」选择视频文件'}</div>
              </div>
            )}
            {isRunning && (
              <>
                <div className="video-overlay">
                  <div className="overlay-tag green">● LIVE</div>
                  <div className="overlay-tag amber">FPS: 30</div>
                </div>
                <div className="video-bottom-bar">
                  <div className="bottom-bar-item"><div className="lbl">当前动作</div><div className="val" style={{ color: 'var(--cyan)' }}>{status.current_pred}</div></div>
                  <div className="bottom-bar-item"><div className="lbl">期望动作</div><div className="val" style={{ color: 'var(--amber)' }}>{status.expected_show}</div></div>
                  <div className="bottom-bar-item"><div className="lbl">帧号</div><div className="val mono" style={{ color: 'var(--text-muted)' }}>{status.frame_id}</div></div>
                  <div className="bottom-bar-item"><div className="lbl">时间</div><div className="val mono" style={{ color: 'var(--text-muted)' }}>{status.time_sec.toFixed(2)}s</div></div>
                  <div className="bottom-bar-item"><div className="lbl">命中</div><div className="val mono" style={{ color: 'var(--green)' }}>{status.hit}/{status.confirm_frames}</div></div>
                </div>
              </>
            )}
          </div>

          <div className="steps-section">
            {STEPS.map((step, i) => (
              <StepCard key={step.id} step={step} index={i} status={stepStatuses[i]} confidence={status.expected_conf} />
            ))}
          </div>
        </div>

        <div className="right-col">
          <div className="panel">
            <div className="panel-title">生产周期</div>
            <div className="cycle-display">
              <div><div className="cycle-big">{status.cycle}</div><div className="cycle-label">当前轮次</div></div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 24, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'var(--green)' }}>{status.cycle > 1 ? status.cycle - 1 : 0}</div>
                <div className="cycle-label">已完成</div>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">实时推理</div>
            <div className="metric-grid">
              <MetricBox label="当前动作" value={status.current_pred} color="cyan" />
              <MetricBox label="期望动作" value={status.expected_show} color="amber" />
              <MetricBox label="置信度" value={`${(status.expected_conf * 100).toFixed(1)}%`} color={status.expected_conf > 0.5 ? 'green' : 'amber'} />
              <MetricBox label="命中帧" value={`${status.hit}/${status.confirm_frames}`} />
            </div>
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Top3 候选</div>
              <div className="top3-list">
                {status.top3 && status.top3 !== '-' ? status.top3.split(' | ').map((item, i) => {
                  const [name, prob] = item.split(':')
                  return (
                    <div className="top3-item" key={i}>
                      <span className="name">{name}</span>
                      <span className={`prob ${parseFloat(prob) > 0.3 ? 'high' : ''}`}>{prob}</span>
                    </div>
                  )
                }) : <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>等待数据...</div>}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">生产统计</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <PassRateRing rate={passRate} />
              <div style={{ flex: 1 }}>
                <div className="stat-row"><span className="stat-lbl">合格</span><span className="stat-val" style={{ color: 'var(--green)' }}>{passCount}</span></div>
                <div className="stat-row"><span className="stat-lbl">超时跳过</span><span className="stat-val" style={{ color: 'var(--red)' }}>{skipCount}</span></div>
                <div className="stat-row"><span className="stat-lbl">总动作数</span><span className="stat-val">{passCount + skipCount}</span></div>
              </div>
            </div>
          </div>

          <div className="panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div className="panel-title">事件日志</div>
            <div className="event-log">
              {events.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>暂无事件</div>
              ) : (
                events.map((ev, i) => (
                  <div className="event-item" key={i}>
                    <div className={`event-icon ${ev.type}`}>{ev.type === 'done' ? '✓' : '!'}</div>
                    <div className="event-info">
                      <div className="event-action">{ev.step.name}</div>
                      <div className="event-time">{ev.timestamp} · {ev.time?.toFixed(1)}s</div>
                    </div>
                    <div className="event-cycle">C{ev.cycle}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="controls">
        <input ref={fileInputRef} type="file" accept="video/*" style={{ display: 'none' }} onChange={handleUpload} />
        <button className="btn" onClick={() => fileInputRef.current?.click()}>
          📁 导入视频{videoName ? `: ${videoName}` : ''}
        </button>
        <button className={`btn primary ${isRunning ? 'danger' : ''}`} onClick={handleStart} disabled={!videoName}>
          {isRunning ? '⏹ 停止' : '▶ 开始分析'}
        </button>
        <div className="spacer"></div>
        <div className="param-input"><label>YOLO阈值</label><input type="text" value={yoloConf} onChange={e => setYoloConf(e.target.value)} /></div>
        <div className="param-input"><label>LSTM阈值</label><input type="text" value={lstmConf} onChange={e => setLstmConf(e.target.value)} /></div>
      </div>
    </div>
  )
}
