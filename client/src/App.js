import { BrowserRouter, Routes, Route } from "react-router-dom";
import { PrimeReactProvider } from 'primereact/api';
import HomePage from "./pages/HomePage";
import './App.css';
import MonitorPage from "./pages/MonitorPage";

function App() {
  return (
    <PrimeReactProvider>
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="Monitor" element={<MonitorPage/>}/>
          <Route path="Home" element={<HomePage />} />
          <Route path="*" element={<HomePage />} />
        </Routes>
      </BrowserRouter>
      
    </div>
    </PrimeReactProvider>
  );
}

export default App;
