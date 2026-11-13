import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import { Article, Note, PageHeader, Section, Stat, StatRow, Tex, articleStyles as a, type TocEntry } from "../components/Article";

const TOC: TocEntry[] = [
  { id: "idea", num: "0.1", title: "The idea" },
  { id: "state", num: "0.2", title: "State of the project" },
  { id: "numbers", num: "0.3", title: "The numbers that matter" },
  { id: "prior-work", num: "0.4", title: "What already exists" },
  { id: "roadmap", num: "0.5", title: "Path to a wearable" },
];

type Row = { stage: string; status: "done" | "in progress" | "next" | "planned"; result: ReactNode; link: string; page: string };

const STATE: Row[] = [
  { stage: "Physics simulator (optics, photorefraction, sensor)", status: "done", result: <>generates unlimited labeled frames; validated against a physical eye</>, link: "/simulation", page: "Simulation" },
  { stage: "PupilNet trained on simulation only", status: "done", result: <>0.65 D on held-out synthetic frames; 152k parameters</>, link: "/network", page: "Network" },
  { stage: "Model-eye campaign (170 settings, 3,400 shots)", status: "done", result: <>0.20 D at 37 cm with a +4/+5 D lens; repeats to ~0.2 D</>, link: "/model-eye", page: "Model eye" },
  { stage: "Failure analysis and self-check", status: "done", result: <>every &gt;0.5 D error traced; lens-step self-check flags bad blocks</>, link: "/network", page: "Network" },
  { stage: "Closed-loop control vs accommodation (RL)", status: "done", result: <>0.46 D vs 0.55 D for the best hand rule &mdash; in simulation</>, link: "/control", page: "Control" },
  { stage: "Video photorefraction (accommodation tracing)", status: "in progress", result: <>~13 reading pairs/s in a synthetic end-to-end test; untested on the Pi</>, link: "/control", page: "Control" },
  { stage: "Strong-lens and close-distance accuracy", status: "in progress", result: <>+8 D lens reads ~1 D weak; unexplained</>, link: "/network", page: "Network" },
  { stage: "Clip-on module (relay + camera + motor)", status: "next", result: <>design modelled; optics checked in simulation; parts list drawn up</>, link: "/prototype", page: "Prototype" },
  { stage: "Alvarez plates characterized at 850 nm", status: "next", result: <>power-per-mm, astigmatism, transmission, glints</>, link: "/alvarez", page: "Alvarez lens" },
  { stage: "Own-eye accommodation trace; fit the patient model", status: "next", result: <>decides whether the learned policy's gain is real</>, link: "/control", page: "Control" },
  { stage: "Assembled prototype on the model eye, then human eyes", status: "planned", result: <>against a clinical reference, under research oversight</>, link: "/prototype", page: "Prototype" },
];

const STATUS_COLOR: Record<Row["status"], string> = {
  done: "var(--green)", "in progress": "var(--blue)", next: "var(--ink)", planned: "var(--ink-3)",
};

export function Overview() {
  return (
    <>
      <PageHeader
        kicker="Overview"
        title="Glasses that measure a prescription and correct it themselves"
        lede={
          <>
            eyesim is a research project on autonomous refraction. An infrared camera photographs the
            pupil, a neural network trained only in simulation reads the refractive error, and an
            adjustable Alvarez lens is set to cancel it. The estimator is now validated on a physical
            model eye; this site documents the physics, the measurements, the control problem and the
            instrument being built around them.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="idea" num="0.1" title="The idea">
          <p>
            A normal eye exam is a slow search guided by the patient's answers. Eccentric photorefraction
            replaces it with one photograph: a small infrared LED sits beside a camera lens, and the pupil
            in the resulting photo shows a bright crescent whose side and extent encode the eye's
            refractive error.
          </p>
          <p>The device is a loop of three steps:</p>
          <ol>
            <li><strong>Sense.</strong> Capture infrared pupil frames, one per LED.</li>
            <li><strong>Estimate.</strong> A small CNN maps the frames to the refraction.</li>
            <li><strong>Correct.</strong> Slide an Alvarez lens to the opposite power &mdash; and re-measure through it until the residual is zero.</li>
          </ol>
          <p>
            The re-measuring matters: because the camera looks through the very lens it is setting, a
            cheap, sticky actuator still converges. Accuracy lives in the sensing loop, not in the
            mechanics. That is the design bet that makes a low-cost wearable plausible.
          </p>
        </Section>

        <Section id="state" num="0.2" title="State of the project">
          <p>
            What has been done, what it produced, and what comes next. Every number is measured, not
            projected; each row links to the page with the full account.
          </p>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Stage</th><th>Status</th><th>Result</th><th></th></tr></thead>
              <tbody>
                {STATE.map((r) => (
                  <tr key={r.stage}>
                    <td>{r.stage}</td>
                    <td style={{ whiteSpace: "nowrap", color: STATUS_COLOR[r.status] }}>{r.status}</td>
                    <td>{r.result}</td>
                    <td><Link to={r.link}>{r.page}</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="numbers" num="0.3" title="The numbers that matter">
          <StatRow>
            <Stat value="0.20" unit="D" label="mean |error| on a physical model eye at 37 cm with a +4/+5 D trial lens (26 settings), network trained on zero real photos" />
            <Stat value="0.26" unit="D" label="mean |error| after removing and re-placing the eye (repeatability)" />
            <Stat value="0.46 vs 0.55" unit="D" label="learned control policy vs the best hand-written rule, on 2,000 simulated refocusing eyes" />
            <Stat value="0.27" unit="D" label="cost of a 0.5 mm pupil mis-centring — why centring is a first-class problem" />
          </StatRow>
          <Note label="Read the numbers honestly.">
            <p>
              The 0.20 D holds at the design distance with a moderate lens; a +8 D lens or a 25&ndash;30 cm
              distance degrades it to 0.66&ndash;1.0 D, and only settings whose photo shows a crescent are
              measurable at all (63 of 170 &mdash; the rest sit in the physical blind band, which the trial
              lens exists to escape). The control result is simulation-only. Both caveats have measurement
              plans, not excuses: see <Link to="/network">Network</Link> and <Link to="/control">Control</Link>.
            </p>
          </Note>
        </Section>

        <Section id="prior-work" num="0.4" title="What already exists">
          <p className={a.runin}>
            <strong>Reading crescents with networks</strong> exists: on real photographs, against clinical
            references (Chun 2020; Xu 2022, 0.18 D with six LED meridians at 1 m; Vu/Essilor 2024, trained
            on simulation <em>plus</em> real volunteer images). Training on simulation alone and validating
            against an exactly known physical eye &mdash; with each failure explained &mdash; is our corner.
          </p>
          <p className={a.runin}>
            <strong>Tunable-lens refraction instruments</strong> exist as benchtop and handheld devices
            (e.g. a wavefront-sensor instrument with a tunable lens, PlenOptika's patent family). A clip-on
            that measures <em>through</em> the wearer's own adjustable lens and then sets it: not found.
          </p>
          <p className={a.runin}>
            <strong>Control against accommodation</strong> is handled clinically by drops or fixed fogging
            protocols. A learned, per-person policy: not found. Closest prior art is RL for other eye
            instruments (perimetry ordering; OCT focus).
          </p>
          <p>
            Searches were web-scale, not a patent clearance. The <Tex>M</Tex>&middot;<Tex>D</Tex> ambiguity
            and the sim-to-real gap called out on this page's first version are both now measured on the
            bench instead of feared: see <Link to="/model-eye">Model eye</Link>.
          </p>
        </Section>

        <Section id="roadmap" num="0.5" title="Path to a wearable">
          <ol>
            <li><strong>Simulate the full loop</strong> &mdash; done, with measured results.</li>
            <li><strong>Validate the estimator on a physical model eye</strong> &mdash; done: 0.20 D at the design point; weak spots identified and listed.</li>
            <li><strong>Measure a real eye's accommodation trace</strong> with the new video mode; fit the patient model; settle whether the learned policy survives contact with a human eye.</li>
            <li><strong>Characterize the Alvarez plates</strong> at 850 nm and motorize one pair.</li>
            <li><strong>Build the clip-on module</strong> around the relay lens and run the whole loop on the model eye.</li>
            <li><strong>Human validation</strong> against a clinical reference, under research oversight.</li>
          </ol>
          <p>
            The <Link to="/tools/simulate">Tools</Link> section still runs the full-resolution physics and
            the trained checkpoint on your own images, from the browser.
          </p>
        </Section>
      </Article>
    </>
  );
}
