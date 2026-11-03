import { Link } from "react-router-dom";
import { Article, Figure, Note, PageHeader, Section, Tex, articleStyles as a, type TocEntry } from "../components/Article";
import { BenchFigure, CaptureFigure, VergenceFigure } from "../figures/BenchFigures";

const TOC: TocEntry[] = [
  { id: "why", num: "2.1", title: "Why build an eye" },
  { id: "bench", num: "2.2", title: "The bench" },
  { id: "physics", num: "2.3", title: "Sphere from screen travel" },
  { id: "builds", num: "2.4", title: "Build path" },
  { id: "calibration", num: "2.5", title: "Calibration" },
  { id: "data", num: "2.6", title: "Data capture" },
  { id: "safety", num: "2.7", title: "IR safety" },
];

export function ModelEye() {
  return (
    <>
      <PageHeader
        kicker="Part 2: hardware"
        title="A motorized model eye with a known prescription"
        lede={
          <>
            The simulator can generate unlimited training images, but it can't tell us how far real images
            differ from synthetic ones. We are building a benchtop eye that can be set to hundreds of
            known prescriptions automatically. It supplies labeled real photorefraction frames for
            fine-tuning, and a physical plant for testing the control loop.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="why" num="2.1" title="Why build an eye">
          <p>
            Fine-tuning needs thousands of real frames, each with a trustworthy <Tex>(S, C, \theta)</Tex>.
            Human subjects can't supply that cheaply, since every frame would need a clinical
            autorefraction. A commercial model eye is certified for exactly one prescription. What we need
            instead is <strong>one rig that can be set to many known prescriptions, repeatably and
            unattended</strong>, at the scale of the 100,000-frame synthetic set.
          </p>
          <ul>
            <li>Sphere from −6 to +4 D and cylinder from 0 to −3 D at any axis, the same range as the synthetic sampler.</li>
            <li>A diffuse, near-IR-reflective "retina" behind an adjustable pupil.</li>
            <li>Rigid, light-sealed mounting, so that repeated measurements actually repeat.</li>
            <li>An exact label for every configuration, produced by script.</li>
          </ul>
        </Section>

        <Section id="bench" num="2.2" title="The bench">
          <p>
            The rig mirrors the most common cause of real myopia. A fixed "cornea" lens focuses
            light, and the retina sits too far behind it (or too close). A stepper-driven lead screw moves
            a PTFE diffuser along the optical axis to set the sphere. A cylindrical trial lens on a
            motorized rotation stage in front of the pupil adds astigmatism at any axis. An IR camera with
            two eccentric 850 nm LEDs images the pupil exactly as the wearable will.
          </p>
          <Figure
            num={1}
            caption={
              <>
                Interactive model of the motorized bench (Option B). Commanding a prescription drives the
                screen carriage and rotation stage. The blue disc on the screen is the geometric blur spot.
                The insets show the photorefraction frames the simulator predicts for this configuration
                at the bench's 300 mm working distance. Stepper counts assume a T8 lead screw and 1/16
                microstepping.
              </>
            }
          >
            <BenchFigure />
          </Figure>
        </Section>

        <Section id="physics" num="2.3" title="Sphere from screen travel">
          <p>
            Put the screen exactly one focal length <Tex>f_0</Tex> behind the lens and the eye is
            emmetropic by construction. Move it by <Tex>\Delta x</Tex>. The correcting power needed at the
            cornea to refocus on the new screen position follows from thin-lens vergence:
          </p>
          <Tex block>{String.raw`S \;=\; \frac{1}{f_0 + \Delta x} - \frac{1}{f_0} \;\approx\; -\frac{\Delta x}{f_0^{2}}`}</Tex>
          <p>
            Moving the screen away from the lens gives myopia (negative <Tex>S</Tex>), which matches the
            sign convention used throughout eyesim. The photorefraction physics depends only on the
            emergent wavefront at the pupil, not on how the dioptres are produced inside the eye. So{" "}
            <Tex>f_0</Tex> can be chosen for mechanical convenience rather than anatomical fidelity.
          </p>
          <Figure num={2} caption="Vergence geometry and the design freedom it gives. With an eye-scale 17 mm lens, 1 D needs only about 0.29 mm of travel, which is a hard precision-machining target. A 50 mm lens needs about 2.5 mm per dioptre, well within reach of a hobby-grade rail.">
            <VergenceFigure />
          </Figure>
          <p>
            Astigmatism composes the same way. A cylindrical trial lens of power <Tex>C</Tex>, rotated to{" "}
            <Tex>\theta</Tex> in contact with the pupil, adds its power directly. The rig's realised
            prescription is <Tex>{String.raw`(S_{\text{screen}},\, C_{\text{lens}},\, \theta_{\text{stage}})`}</Tex>,
            and it feeds straight into <code>prescription_to_power_vector()</code>.
          </p>
        </Section>

        <Section id="builds" num="2.4" title="Build path">
          <p>The three options are sequential, not exclusive. Option A produces a first real dataset while B is being built.</p>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Option</th><th>Mechanism</th><th>Effort</th><th className={a.num}>Cost</th></tr></thead>
              <tbody>
                <tr><td>A</td><td>Trial lenses swapped by hand</td><td>days</td><td className={a.num}>$100 to $150</td></tr>
                <tr><td>B</td><td>Motorized rail and rotation stage</td><td>weeks</td><td className={a.num}>$275 to $650</td></tr>
                <tr><td>C</td><td>Surplus phoropter</td><td>opportunistic</td><td className={a.num}>varies</td></tr>
              </tbody>
            </table>
          </div>
          <p className={a.runin}>
            <strong>A, manual.</strong> The lens and screen stay fixed at 0 D, and optometric trial lenses
            are swapped by hand in front of the pupil. It needs no custom mechanics and yields 100 to 300
            labeled frames per session, enough for a first measurement of the sim-to-real gap.
          </p>
          <p className={a.runin}>
            <strong>B, motorized.</strong> This is the primary build. An MGN12 rail with a T8 lead screw
            sets the sphere, and a stepper-driven rotation mount sets the cylinder axis. Any (S, C, θ)
            can be scripted, so sweeps run unattended overnight, and the same rig later serves as a
            physical plant for the optimizer.
          </p>
          <p className={a.runin}>
            <strong>C, phoropter.</strong> A decommissioned phoropter already contains calibrated sphere,
            cylinder and axis wheels, and some digital units have motors and a serial port. It is worth
            asking the department before building B, which remains the fallback.
          </p>
          <p>
            Precision matters only at the sliding interface. The rail and lead screw are off-the-shelf parts
            with sub-10 µm repeatability, far better than the 2.5 µm ≈ 0.001 D per microstep the geometry
            needs. 3D-printed parts are fine for brackets and for the rotation stage, where backlash is
            much more forgiving.
          </p>
        </Section>

        <Section id="calibration" num="2.5" title="Calibration">
          <p>A label can only be trusted if the rig's zero point and lens specs are trusted. Before any large run:</p>
          <ol>
            <li><strong>Measure the focal length</strong> <Tex>f_0</Tex> by autocollimation rather than trusting the datasheet. The <Tex>1/f_0^2</Tex> term amplifies small errors.</li>
            <li>
              <strong>Find the emmetropic zero empirically.</strong> Put a USAF 1951 chart or eyesim's own
              Siemens star at the screen and sweep the carriage while maximising{" "}
              <code>eyesim.optimize.sharpness()</code> on the live camera feed. The rig calibrates itself
              with the same code that drives the closed loop.
            </li>
            <li><strong>Verify the rotation stage's zero angle</strong> against a printed protractor or an alignment jig.</li>
            <li><strong>Re-check the zero periodically</strong> for thermal drift and mechanical settling. It is the same cheap sharpness sweep.</li>
          </ol>
        </Section>

        <Section id="data" num="2.6" title="Data capture">
          <p>
            Each configuration runs the same short script and writes directly into the synthetic
            dataset's <code>.npz</code> schema, together with the measured geometry (<Tex>D</Tex>,
            eccentricity, pupil) as metadata. Recording <Tex>D</Tex> matters. It is the missing variable
            behind the <Link to="/simulation#photorefraction">M·D degeneracy</Link> in the simulator.
          </p>
          <Figure num={3} caption="Per-configuration capture loop. Click a step to inspect it.">
            <CaptureFigure />
          </Figure>
          <p>
            Fine-tuning starts from <code>checkpoints/best.pt</code> at a reduced learning rate, on batches
            that mix bench and synthetic frames so the network does not forget the larger synthetic set.
            The headline metric is the change in error on held-out <em>real</em> frames, which directly
            measures the sim-to-real gap and how much of it closes.
          </p>
          <p>
            The diffuser is the one known physical gap. A real fundus is partly retro-reflective because
            photoreceptors act as waveguides (the source of red-eye), and a flat PTFE sheet is not. If
            the captured crescents look much lower in contrast than published photorefraction images,
            this is the first thing to revisit.
          </p>
          <Note label="Second use.">
            <p>
              Once motorized, the rail and rotation stage are a physical actuator with a known
              dioptre-per-step relationship. <code>run_closed_loop()</code> can drive them against real camera
              sharpness, testing the control algorithm against real backlash, noise and vibration
              before the <Link to="/alvarez">Alvarez actuator</Link> exists.
            </p>
          </Note>
        </Section>

        <Section id="safety" num="2.7" title="IR safety">
          <p>
            The rig only images a diffuser, but its LEDs and mounts will carry over to human trials, so it
            is designed within safe limits from the start. We use LEDs with a published radiant-intensity
            spec and drive currents of a few to tens of mA. Before any human imaging, exposure will be
            checked against IEC 62471. Infrared is invisible, so the blink reflex offers no protection.
          </p>
        </Section>
      </Article>
    </>
  );
}
