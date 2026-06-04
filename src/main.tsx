import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Note: StrictMode is intentionally omitted. Its dev-only double-invocation of
// effects complicates GPU device/buffer lifecycle; init is already idempotent.
const root = document.getElementById('root');
if (!root) throw new Error('#root not found');
createRoot(root).render(<App />);
