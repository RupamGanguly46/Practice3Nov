// Step 2 work by Poorvi
// Added buttons to call APIs (placeholders)
import React, { useState } from 'react'

export default function App(){
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)

  const callScratch = async ()=>{
    setResult({status: 'simulated', model: 'scratch'})
  }

  return (
    <div style={{ padding: 24, fontFamily: 'Arial' }}>
      <h1>Text Classifier</h1>
      <textarea rows={6} cols={80} value={text} onChange={e=>setText(e.target.value)} placeholder='Type text to classify' />
      <div style={{ marginTop: 12 }}>
        <button onClick={callScratch} disabled={!text}>Predict (Scratch Model)</button>
      </div>
      <div style={{ marginTop: 18 }}>
        <h3>Result</h3>
        { result ? <pre>{JSON.stringify(result, null, 2)}</pre> : <div>No result yet</div> }
      </div>
    </div>
  )
}
