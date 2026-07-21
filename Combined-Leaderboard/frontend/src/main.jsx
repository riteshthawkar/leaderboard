import React from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/geist";
import "./tailwind.css";
import "./design-system.css";
import "./custom-visuals.css";
import App from "./App";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
