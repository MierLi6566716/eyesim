import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { NavBar } from "./NavBar";
import styles from "./Layout.module.css";

export function Layout() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (!hash) window.scrollTo({ top: 0 });
  }, [pathname, hash]);

  return (
    <div className={styles.shell}>
      <NavBar />
      <main className={styles.main}>
        <Outlet />
      </main>
      <footer className={styles.footer}>
        <p>eyesim is a research prototype, not a medical device. Nothing on this site is a clinical measurement.</p>
      </footer>
    </div>
  );
}
