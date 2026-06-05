import { createRoot } from 'react-dom/client';
import App from './App';

import '@fontsource/archivo/600.css';
import '@fontsource/archivo/700.css';
import '@fontsource/archivo/800.css';
import '@fontsource/archivo/900.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/jetbrains-mono/700.css';
import './index.css';

const root = document.getElementById('root');
if (!root) throw new Error('#root not found');
createRoot(root).render(<App />);
