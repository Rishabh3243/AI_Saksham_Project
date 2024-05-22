import { BrowserRouter, Routes, Route } from "react-router-dom";
import { PrimeReactProvider } from 'primereact/api';
import HomePage from "./pages/HomePage";
import VideoToLanguagePage from "./pages/VideoToLanguagePage";
import './App.css';

function App() {
  return (
    <PrimeReactProvider>
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="VideoToLanguage" element={<VideoToLanguagePage/>}/>
          <Route path="*" element={<HomePage />} />
        </Routes>
      </BrowserRouter>
      
    </div>
    </PrimeReactProvider>
  );
}

export default App;
