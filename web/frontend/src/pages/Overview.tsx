import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import { Article, Note, PageHeader, Section, Stat, StatRow, articleStyles as a, type TocEntry } from "../components/Article";

const TOC: TocEntry[] = [
  { id: "idea", num: "0.1", title: "Concept" },
  { id: "state", num: "0.2", title: "Project status" },
  { id: "results", num: "0.3", title: "Principal results" },
  { id: "prior-work", num: "0.4", title: "Related work" },
  { id: "roadmap", num: "0.5", title: "Development plan" },
];

type Row = { stage: string; status: "complete" | "in progress" | "next" | "planned"; result: ReactNode; link: string; page: string };

const STATE: Row[] = [
  { stage: "Physics simulator (optics, photorefraction, sensor)", status: "complete", result: <>labelled training images; compared against a physical eye</>, link: "/simulation", page: "Simulation" },
  { stage: "Refraction network trained on simulated images", status: "complete", result: <>152,451 parameters; 0.65 D on held-out simulated images</>, link: "/network", page: "Network" },
  { stage: "Model-eye validation (170 settings, 3,400 shots)", status: "complete", result: <>0.20 D at 37 cm with a +4 or +5 D lens; repeatable to about 0.2 D</>, link: "/model-eye", page: "Model eye" },
  { stage: "Error analysis and consistency check", status: "complete", result: <>sources of errors above 0.5 D identified; lens-step check flags unreliable blocks</>, link: "/network", page: "Network" },
  { stage: "Learned control under accommodation", status: "complete", result: <>0.46 D against 0.55 D for the best fixed procedure, in simulation</>, link: "/control", page: "Control" },
  { stage: "Human-eye measurement", status: "in progress", result: <>network under-reads by 1 to 5.6 D; LED balance and capture conditions under correction</>, link: "/network", page: "Network" },
  { stage: "Video photorefraction", status: "in progress", result: <>about 13 frame pairs per second in a synthetic test; not yet run on the camera</>, link: "/control", page: "Control" },
  { stage: "Accuracy with strong trial lenses and at short distance", status: "in progress", result: <>+8 D lens reads about 1 D weak; cause not yet identified</>, link: "/network", page: "Network" },
  { stage: "Clip-on measurement module", status: "next", result: <>design modelled; relay optics verified in simulation</>, link: "/prototype", page: "Prototype" },
  { stage: "Alvarez plate characterisation at 850 nm", status: "next", result: <>power per millimetre, astigmatism, transmission, reflections</>, link: "/alvarez", page: "Alvarez lens" },
  { stage: "Integrated prototype on the model eye, then on human eyes", status: "planned", result: <>against a clinical reference, under research oversight</>, link: "/prototype", page: "Prototype" },
];

const STATUS_COLOR: Record<Row["status"], string> = {
  complete: "var(--green)", "in progress": "var(--blue)", next: "var(--ink)", planned: "var(--ink-3)",
};

export function Overview() {
  return (
    <>
      <PageHeader
        kicker="Overview"
        title="Glasses that measure a prescription and correct it themselves"
        lede={
          <>
            eyesim is a research project on automatic refraction. An infrared camera photographs the
            pupil, a neural network trained on simulated images estimates the refractive error, and an
            adjustable Alvarez lens is set to correct it. The estimator has been validated on a physical
            model eye; work on human eyes and on the wearable instrument is in progress.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="idea" num="0.1" title="Concept">
          <p>
            A conventional eye examination is an iterative search guided by the patient's responses.
            Eccentric photorefraction replaces it with a photograph: an infrared LED placed beside a
            camera lens produces a bright crescent in the pupil, and the side and extent of that crescent
            depend on the refractive error of the eye.
          </p>
          <p>The instrument repeats three steps:</p>
          <ol>
            <li><strong>Sense.</strong> Record infrared pupil frames, one for each LED.</li>
            <li><strong>Estimate.</strong> A small convolutional network converts the frames into a refraction.</li>
            <li><strong>Correct.</strong> Move an Alvarez lens to the opposite power, then measure again through it until the residual error is zero.</li>
          </ol>
          <p>
            Because the camera views the eye through the lens being adjusted, each measurement reports
            the remaining error directly. An inexpensive actuator with backlash still converges; accuracy
            is set by the measurement, not by the mechanism. This is the basis for a low-cost wearable.
          </p>
        </Section>

        <Section id="state" num="0.2" title="Project status">
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

        <Section id="results" num="0.3" title="Principal results">
          <StatRow>
            <Stat value="0.20" unit="D" label="mean absolute error on a physical model eye at 37 cm with a +4 or +5 D trial lens (26 settings), with no real images in training" />
            <Stat value="0.26" unit="D" label="mean absolute error after the eye was removed and replaced" />
            <Stat value="0.46" unit="D" label="learned lens control on 2,000 simulated eyes, against 0.55 D for the best fixed procedure" />
            <Stat value="0.27" unit="D" label="median change in reading per 0.5 mm of pupil-centring error" />
          </StatRow>
          <Note label="Conditions.">
            <p>
              The 0.20 D figure applies at the design distance with a moderate trial lens. With a +8 D
              lens, or at 25 to 30 cm, the error rises to 0.66 to 1.0 D, and only settings whose
              photograph shows a crescent can be measured (63 of 170). On a human eye the network
              currently under-reads by 1 to 5.6 D. The control result is from simulation.
            </p>
          </Note>
        </Section>

        <Section id="prior-work" num="0.4" title="Related work">
          <p className={a.runin}>
            <strong>Network-based photorefraction.</strong> Networks trained on real photographs have been
            evaluated against clinical refraction (Chun et al., 2020; Xu et al., 2022, 0.18 D with six LED
            meridians at 1 m). Vu et al. (2024) trained on simulated and real volunteer images combined.
            This project trains on simulated images only and validates against an eye of exactly known
            refraction.
          </p>
          <p className={a.runin}>
            <strong>Tunable-lens refraction.</strong> Bench-top and hand-held instruments combine a
            tunable lens with a wavefront sensor (for example US patent 11,096,576). No clip-on device
            that measures through the wearer's own adjustable lens and then sets it was found.
          </p>
          <p className={a.runin}>
            <strong>Accommodation control.</strong> Clinical practice uses cycloplegic drops or a fixed
            fogging protocol. No learned, individually adaptive procedure was found; the nearest
            reinforcement-learning work in ophthalmic instruments concerns perimetry and OCT focusing.
          </p>
        </Section>

        <Section id="roadmap" num="0.5" title="Development plan">
          <ol>
            <li><strong>Simulate the complete loop.</strong> Complete.</li>
            <li><strong>Validate the estimator on a physical model eye.</strong> Complete; limitations identified.</li>
            <li><strong>Repeat the human-eye measurement</strong> with balanced LEDs, and record the time course of accommodation with the video mode.</li>
            <li><strong>Characterise the Alvarez plates</strong> at 850 nm and motorise one pair.</li>
            <li><strong>Build the clip-on module</strong> and operate the full loop on the model eye.</li>
            <li><strong>Evaluate on human eyes</strong> against a clinical reference, under research oversight.</li>
          </ol>
        </Section>
      </Article>
    </>
  );
}
