import React from "react";
import { createRoot } from "react-dom/client";
import "./tailwind.css";
import "../static/css/site.css";
import "./react.css";
import App from "./App";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);