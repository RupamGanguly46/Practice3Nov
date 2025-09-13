// Step 4 work by Poorvi
// UI tweaks and notes section
import React, { useState } from 'react'
import { predictScratch, predictHF, trainScratch } from './api'

export default function App(){
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const callScratch = async () => {
    setLoading(true)
    try{
      const res = await predictScratch(text)
      setResult(res.data)
    }catch(e){
      setResult({ error: e.message })
    }finally{
      setLoading(false)
    }
  }

  const callHF = async () => {
    setLoading(true)
    try{
      const res = await predictHF(text)
      setResult(res.data)
    }catch(e){
      setResult({ error: e.message })
    }finally{
      setLoading(false)
    }
  }

  const retrain = async () => {
    try{
      await trainScratch()
      alert('Retrain triggered. Check backend logs.')
    }catch(e){
      alert('Retrain failed: ' + e.message)
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'Arial' }}>
      <h1>Text Classifier</h1>
      <textarea rows={6} cols={80} value={text} onChange={e=>setText(e.target.value)} placeholder='Type text to classify' />
      <div style={{ marginTop: 12 }}>
        <button onClick={callScratch} disabled={!text}>Predict (Scratch Model)</button>
        <button onClick={callHF} disabled={!text} style={{ marginLeft: 8 }}>Predict (HuggingFace)</button>
        <button onClick={retrain} style={{ marginLeft: 8 }}>Retrain Scratch Model</button>
      </div>

      <div style={{ marginTop: 18 }}>
        <h3>Result</h3>
        {loading ? <div>Loading...</div> : (
          result ? <pre>{JSON.stringify(result, null, 2)}</pre> : <div>No result yet</div>
        )}
      </div>

      <div style={{ marginTop: 18 }}>
        <h4>Notes</h4>
        <ul>
          <li>Scratch model is small and saved at <code>backend/models/scratch_model.joblib</code>.</li>
          <li>Hugging Face model downloads automatically when first used; make sure the backend has internet access.</li>
        </ul>
      </div>
    </div>
  )
}
