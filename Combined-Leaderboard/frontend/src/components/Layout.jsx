import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { Menu, Moon, Sun, X } from "lucide-react";
import { Button } from "@/components/ui/button";

const nav = [
  ["/", "Home", "home"],
  ["/benchmarks/do-you-see-me", "Do-You-See-Me", "dysm"],
  ["/benchmarks/minds-eye", "Mind's-Eye", "minds_eye"],
  ["/benchmarks/spatial", "Spatial", "spatial"],
  ["/leaderboard", "Leaderboard", "leaderboard"],
];

function pageId(pathname) {
  if (pathname === "/") return "home";
  if (pathname.includes("do-you-see-me")) return "dysm";
  if (pathname.includes("minds-eye")) return "minds_eye";
  if (pathname.includes("spatial")) return "spatial";
  if (pathname.includes("leaderboard")) return "leaderboard";
  if (pathname.includes("submit")) return "submit";
  if (pathname.includes("login")) return "login";
  return "";
}

export function Layout() {
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem("vci-theme") || "dark");
  const [online, setOnline] = useState(false);

  useEffect(() => {
    document.body.setAttribute("data-page", pageId(location.pathname));
    setOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("vci-theme", theme);
    window.dispatchEvent(new Event("themechange"));
  }, [theme]);

  useEffect(() => {
    let live = true;
    const check = () => fetch("/api/health").then((r) => live && setOnline(r.ok)).catch(() => live && setOnline(false));
    check();
    const id = setInterval(check, 30000);
    return () => { live = false; clearInterval(id); };
  }, []);

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="container header-inner">
          <Link to="/" className="brand" aria-label="Home">
            <span className="brand-text"><strong>VISTA</strong></span>
          </Link>
          <nav className={`header-nav ${open ? "open" : ""}`} id="nav_menu" aria-label="Primary">
            {nav.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `nav-link ${isActive ? "is-active" : ""}`}>
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="header-actions">
            <Button asChild variant="brand" size="sm" className={`submit-nav-cta ${location.pathname.includes("submit") ? "is-active" : ""}`.trim()}>
              <Link to="/submit">Submit a model</Link>
            </Button>
            <button className="icon-btn nav-toggle" type="button" aria-label="Menu" onClick={() => setOpen((value) => !value)}>
              {open ? <X size={18} /> : <Menu size={18} />}
            </button>
          </div>
        </div>
      </header>

      <main><Outlet /></main>

      <footer className="site-footer">
        <div className="container">
          <div className="footer-inner">
            <div className="footer-col" style={{ maxWidth: 320 }}>
              <Link to="/" className="brand" style={{ marginBottom: 12 }}>
                <span className="brand-text"><strong>VISTA</strong></span>
              </Link>
              <p className="muted small">A unified, paper-faithful evaluation of multimodal LLM visual perception, mental imagery, and spatial reasoning across three Microsoft Research benchmarks.</p>
            </div>
            <div className="footer-col"><h5>Benchmarks</h5><Link to="/benchmarks/do-you-see-me">Do-You-See-Me</Link><Link to="/benchmarks/minds-eye">Mind's-Eye</Link><Link to="/benchmarks/spatial">Spatial Reasoning</Link></div>
            <div className="footer-col"><h5>Explore</h5><Link to="/leaderboard">Leaderboard</Link><Link to="/submit">Submit a model</Link><a href="/#findings">Key findings</a></div>
            <div className="footer-col"><h5>Papers</h5><a href="https://arxiv.org/abs/2506.02022" target="_blank" rel="noreferrer">Do You See Me ↗</a><a href="https://arxiv.org/abs/2604.16054" target="_blank" rel="noreferrer">Mind's Eye ↗</a><a href="https://arxiv.org/abs/2604.16060" target="_blank" rel="noreferrer">CoT Degrades Spatial ↗</a></div>
          </div>
          <div className="footer-bottom">
            <span>Benchmarks © Microsoft Research. Leaderboard for non-commercial research use.</span>
            <div className="footer-meta">
              <span className="status-pill" title="Backend status"><span className={`status-dot ${online ? "ok" : ""}`} /><span>{online ? "Online" : "Offline"}</span></span>
              <button className="icon-btn" id="theme_toggle" type="button" aria-label="Toggle dark mode" title="Toggle theme" onClick={() => setTheme((value) => value === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}</button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}