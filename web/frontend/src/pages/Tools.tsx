import { NavLink, Outlet, useLocation, Navigate } from "react-router-dom";
import { API_BASE } from "../lib/api";
import styles from "./Tools.module.css";

const TABS = [
  { to: "/tools/simulate", label: "Simulate a photo" },
  { to: "/tools/diagnose", label: "Diagnose (experimental)" },
  { to: "/tools/gallery", label: "Gallery" },
];

/** Wrapper for the tools that call the FastAPI backend (full-resolution Python physics + PupilNet). */
export function Tools() {
  const { pathname } = useLocation();
  if (pathname === "/tools" || pathname === "/tools/") return <Navigate to="/tools/simulate" replace />;
  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <h1>Tools</h1>
        <p>
          These run on the Python backend (<code>eyesim</code> at full resolution, and the trained
          PupilNet checkpoint) instead of in your browser. They need the API at{" "}
          <code>{API_BASE}</code> to be reachable.
        </p>
        <nav className={styles.tabs}>
          {TABS.map((t) => (
            <NavLink key={t.to} to={t.to} className={({ isActive }) => (isActive ? styles.on : undefined)}>
              {t.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <Outlet />
    </div>
  );
}
