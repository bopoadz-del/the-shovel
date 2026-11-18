import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './theme/tokens.css'
import App from './App.tsx'
import BuildMismatchBanner from './components/BuildMismatchBanner'
import { startBuildCheck } from './lib/buildWatch'

startBuildCheck()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BuildMismatchBanner />
    <App />
  </StrictMode>,
)
