import { Link } from "react-router-dom";
import { Article, Figure, Note, PageHeader, Section, Stat, StatRow, Tex, articleStyles as a, type TocEntry } from "../components/Article";

const TOC: TocEntry[] = [
  { id: "idea", num: "0.1", title: "The idea" },
  { id: "prior-work", num: "0.2", title: "What already exists" },
  { id: "built", num: "0.3", title: "What has been built" },
  { id: "hardware", num: "0.4", title: "What hardware must answer" },
  { id: "roadmap", num: "0.5", title: "Path to a wearable" },
];

export function Overview() {
  return (
    <>
      <PageHeader
        kicker="Overview"
        title="Glasses that measure a prescription and correct it themselves"
        lede={
          <>
            eyesim is a research project on autonomous refraction. An infrared camera photographs the
            pupil, a neural network reads the refractive error from that photo, and a tunable lens is set
            to cancel it. The whole loop runs in simulation today. This site documents that simulation
            and the hardware planned to test it.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="idea" num="0.1" title="The idea">
          <p>
            A normal eye exam is a slow search guided by the patient's answers. Eccentric photorefraction
            replaces it with one measurement. A small infrared LED sits just beside a camera lens. In the
            resulting photo the pupil shows a bright crescent, and the crescent's size, side and shape
            depend on the eye's refractive error in a predictable way.
          </p>
          <p>The device is a loop of three steps:</p>
          <ol>
            <li><strong>Sense.</strong> Capture two infrared pupil frames, one for each of two LEDs.</li>
            <li><strong>Estimate.</strong> A small CNN maps the frames to a power vector <Tex>{"(M, J_0, J_{45})"}</Tex>.</li>
            <li><strong>Correct.</strong> Set a tunable lens to the opposite of that estimate.</li>
          </ol>
          <p>
            One photo gives one answer and one lens move. No trial and error is needed, for the same
            reason a clinical autorefractor asks the patient no questions.
          </p>
          <StatRow>
            <Stat value="0.18" unit="D" label="Mean absolute error on spherical equivalent M (synthetic test set)" />
            <Stat value="0.05" unit="D" label={<>Mean absolute error on <Tex>J_0</Tex> and <Tex>{"J_{45}"}</Tex></>} />
            <Stat value="1.34 → 0.08" unit="D" label="Median residual error across 20 random eyes after one correction" />
            <Stat value="100,000" label="Simulated photorefraction samples (80k train, 10k validation, 10k test)" />
          </StatRow>
          <Note label="Caveat.">
            <p>
              Every figure above comes from simulated images graded against simulated ground truth.
              They show that the pipeline is internally consistent. They do not yet show that it works
              on a real camera and a real eye.
            </p>
          </Note>
        </Section>

        <Section id="prior-work" num="0.2" title="What already exists">
          <p>
            Self-correcting glasses split into three problems. Each has research behind it. Nobody has
            joined the first two into one working loop.
          </p>
          <p className={a.runin}>
            <strong>Measuring automatically.</strong> Eccentric photorefraction is a decades-old clinical
            technique, used in autorefractors. Recent work trains models to read the crescent directly
            instead of using hand-built formulas.
          </p>
          <p className={a.runin}>
            <strong>Tuning a lens electrically.</strong> Liquid-crystal lenses are commercially available,
            at roughly $1,000 per lens, and they need continuous voltage to hold a setting.
          </p>
          <p className={a.runin}>
            <strong>Pre-distorting a display.</strong> An image can be processed so that a blurry eye sees
            it sharp. This only works for that one screen, not for the rest of the room.
          </p>
          <p>
            This project makes two bets. The first is integration: sensing and correction in one device,
            driven by one estimate. The second is the actuator. Instead of liquid crystal we use an{" "}
            <Link to="/alvarez">Alvarez lens</Link>, a 1967 design that is cheap to mould and can hold its
            setting mechanically.
          </p>
        </Section>

        <Section id="built" num="0.3" title="What has been built">
          <p>
            Everything so far is software, with measured results. Each stage has an interactive figure on
            the <Link to="/simulation">Simulation</Link> page.
          </p>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Stage</th><th>What it does</th><th>Section</th></tr></thead>
              <tbody>
                <tr>
                  <td>Wave optics</td>
                  <td>Turns a prescription into a point-spread function and a blurred retinal image.</td>
                  <td><Link to="/simulation#wave-optics">1.2</Link></td>
                </tr>
                <tr>
                  <td>Pre-distortion</td>
                  <td>Runs the same physics in reverse so a known eye sees a display sharply.</td>
                  <td><Link to="/simulation#predistortion">1.3</Link></td>
                </tr>
                <tr>
                  <td>Photorefraction</td>
                  <td>Synthesises the infrared pupil frames the planned sensor would capture.</td>
                  <td><Link to="/simulation#photorefraction">1.4</Link></td>
                </tr>
                <tr>
                  <td>PupilNet</td>
                  <td>A CNN that regresses the power vector from two 64×64 frames.</td>
                  <td><Link to="/simulation#cnn">1.5</Link></td>
                </tr>
                <tr>
                  <td>Closed loop</td>
                  <td>Applies the estimate as a lens command, with optional sharpness refinement.</td>
                  <td><Link to="/simulation#closed-loop">1.6</Link></td>
                </tr>
              </tbody>
            </table>
          </div>
          <Figure
            num={1}
            caption={
              <>
                Residual refractive error for 20 random simulated eyes at each stage of the pipeline. The
                one-shot CNN correction brings 95% of them inside the 0.25 D clinical step. The final
                refinement stage needs a camera that sees what the eye sees, so it applies to the bench
                and not to the wearable.
              </>
            }
          >
            <img
              src="/figures/pipeline_cohort.png"
              alt="Residual error per eye, uncorrected, after the CNN correction and after refinement, with the share of eyes within 0.25 dioptres"
              style={{ width: "100%", display: "block", background: "#fff" }}
            />
          </Figure>
          <p>
            The <Link to="/tools/simulate">Tools</Link> section runs the full-resolution Python physics
            and the trained checkpoint on your own images.
          </p>
        </Section>

        <Section id="hardware" num="0.4" title="What hardware must answer">
          <p>A simulation is only as good as its assumptions. Two limits are already known.</p>
          <ul>
            <li>
              <strong>The M·D ambiguity.</strong> The simulated crescent depends on the spherical error
              and the camera distance only through their product. This is why <Tex>M</Tex> is less
              accurate than the astigmatic components. In a real frame the distance is fixed by design.
            </li>
            <li>
              <strong>The sim-to-real gap.</strong> Real frames have sensor noise, stray reflections,
              misalignment and eyes with aberrations the model leaves out.
            </li>
          </ul>
          <p>
            Two benches address them. The <Link to="/model-eye">model eye</Link> comes first: a motorized
            benchtop eye that can be set to hundreds of known prescriptions. It supplies labeled real
            frames for fine-tuning and a physical plant for the control loop. The{" "}
            <Link to="/alvarez">Alvarez lens</Link> is the corrective element. It has to be built and
            measured to confirm that slide distance maps to optical power as the formula predicts.
          </p>
          <p>Some questions can only be answered once both exist:</p>
          <ul>
            <li>Is the loop fast enough, and is its power draw low enough, for all-day wear?</li>
            <li>Does the sensor still work in sunlight, which is a strong infrared source?</li>
            <li>How does it behave on real eyes that blink, move and change pupil size?</li>
          </ul>
        </Section>

        <Section id="roadmap" num="0.5" title="Path to a wearable">
          <p>Each step is useful on its own. The last one is expected to take longer than all the others combined.</p>
          <ol>
            <li><strong>Simulate the full loop</strong> (done). Optics, sensor, estimator and control, with measured results.</li>
            <li><strong>Build the model-eye bench</strong> (next). Collect real labeled frames and measure the sim-to-real gap.</li>
            <li><strong>Build and measure the Alvarez lens</strong> on its own bench.</li>
            <li><strong>Combine both benches.</strong> This is the first time the whole loop runs on real hardware.</li>
            <li><strong>Miniaturize</strong> the camera, processing, actuator and battery into a frame.</li>
            <li><strong>Validate on human eyes</strong> against a clinical reference, under research oversight.</li>
          </ol>
        </Section>
      </Article>
    </>
  );
}
