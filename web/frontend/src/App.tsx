import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Overview } from "./pages/Overview";
import { Simulation } from "./pages/Simulation";
import { ModelEye } from "./pages/ModelEye";
import { Alvarez } from "./pages/Alvarez";
import { Prototype } from "./pages/Prototype";
import { Network } from "./pages/Network";
import { Control } from "./pages/Control";
import { Tools } from "./pages/Tools";
import { Simulate } from "./pages/Simulate";
import { Diagnose } from "./pages/Diagnose";
import { Gallery } from "./pages/Gallery";
import { Share } from "./pages/Share";

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="simulation" element={<Simulation />} />
        <Route path="model-eye" element={<ModelEye />} />
        <Route path="alvarez" element={<Alvarez />} />
        <Route path="network" element={<Network />} />
        <Route path="control" element={<Control />} />
        <Route path="prototype" element={<Prototype />} />
        <Route path="tools" element={<Tools />}>
          <Route path="simulate" element={<Simulate />} />
          <Route path="diagnose" element={<Diagnose />} />
          <Route path="gallery" element={<Gallery />} />
        </Route>
        <Route path="share/:slug" element={<Share />} />
      </Route>
    </Routes>
  );
}

export default App;
