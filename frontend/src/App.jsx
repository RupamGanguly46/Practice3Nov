// Step 1 work by Poorvi
// Basic React app shell
import React, { useState } from 'react'

export default function App(){
  const [text, setText] = useState('')
  return (
    <div style={{ padding: 24, fontFamily: 'Arial' }}>
      <h1>Text Classifier (step1)</h1>
      <textarea rows={6} cols={80} value={text} onChange={e=>setText(e.target.value)} placeholder='Type text to classify' />
    </div>
  )
}
