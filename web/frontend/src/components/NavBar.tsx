import { NavLink } from "react-router-dom";
import styles from "./NavBar.module.css";

const LINKS = [
  { to: "/", label: "Overview", num: "", end: true },
  { to: "/simulation", label: "Simulation", num: "1" },
  { to: "/model-eye", label: "Model eye", num: "2" },
  { to: "/alvarez", label: "Alvarez lens", num: "3" },
];

export function NavBar() {
  return (
    <header className={styles.bar}>
      <div className={styles.inner}>
        <NavLink to="/" className={styles.brand} aria-label="eyesim home">
          eyesim
        </NavLink>
        <nav className={styles.links}>
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) => `${styles.link} ${isActive ? styles.active : ""}`}
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
        <NavLink to="/tools/simulate" className={({ isActive }) => `${styles.tools} ${isActive ? styles.active : ""}`}>
          Tools
        </NavLink>
      </div>
    </header>
  );
}
