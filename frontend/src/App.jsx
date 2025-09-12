// Step 3 work by Poorvi
// Wire up api.js and add HF button
import React, { useState } from 'react'
import { predictScratch, predictHF } from './api'

export default function App(){
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)

  const callScratch = async () => {
    try{
      const res = await predictScratch(text)
      setResult(res.data)
    }catch(e){
      setResult({error: e.message})
    }
  }

  const callHF = async () => {
    try{
      const res = await predictHF(text)
      setResult(res.data)
    }catch(e){
      setResult({error: e.message})
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'Arial' }}>
      <h1>Text Classifier</h1>
      <textarea rows={6} cols={80} value={text} onChange={e=>setText(e.target.value)} placeholder='Type text to classify' />
      <div style={{ marginTop: 12 }}>
        <button onClick={callScratch} disabled={!text}>Predict (Scratch Model)</button>
        <button onClick={callHF} disabled={!text} style={{ marginLeft: 8 }}>Predict (HuggingFace)</button>
      </div>
      <div style={{ marginTop: 18 }}>
        <h3>Result</h3>
        { result ? <pre>{JSON.stringify(result, null, 2)}</pre> : <div>No result yet</div> }
      </div>
    </div>
  )
}
