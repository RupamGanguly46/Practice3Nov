// Step 3 work by Poorvi
import axios from 'axios'
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000'

export const predictScratch = (text) => axios.post(`${API_BASE}/predict/scratch`, { text })
export const predictHF = (text) => axios.post(`${API_BASE}/predict/hf`, { text })
export const trainScratch = () => axios.post(`${API_BASE}/train`)
