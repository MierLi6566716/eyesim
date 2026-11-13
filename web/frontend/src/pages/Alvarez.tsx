import { Link } from "react-router-dom";
import { Article, Figure, Note, PageHeader, Section, Tex, articleStyles as a, type TocEntry } from "../components/Article";
import { ActuatorFigure, AlvarezLensFigure } from "../figures/AlvarezFigures";
import { ACT, AMPS_PER_DIOPTRE, A_PER_MM2, GAIN_D_PER_MM, N_PMMA } from "../lib/alvarez";

const TOC: TocEntry[] = [
  { id: "why", num: "3.1", title: "Why not liquid crystal" },
  { id: "plates", num: "3.2", title: "Two plates, one lens" },
  { id: "axes", num: "3.3", title: "Covering all three components" },
  { id: "actuator", num: "3.4", title: "Moving the plates" },
  { id: "measure", num: "3.5", title: "What the bench must measure" },
];

const ACT_ROWS: [string, string][] = [
  ["Moving mass (plate + coil)", `${ACT.massKg * 1000} g`],
  ["Flexure stiffness", `${ACT.springNPerM} N/m`],
  ["Damping", `${ACT.dampNsPerM} N·s/m`],
  ["Voice-coil force constant", `${ACT.forceNPerA.toFixed(1)} N/A`],
  ["Coil resistance", `${ACT.coilOhm} Ω`],
  ["Current limit", `${(ACT.iMaxA * 1000).toFixed(0)} mA`],
  ["Position-sensor noise (1σ)", `${(ACT.hallNoiseM * 1e6).toFixed(1)} µm`],
];

export function Alvarez() {
  return (
    <>
      <PageHeader
        kicker="Part 3: actuator"
        title="An Alvarez lens as the corrective element"
        lede={
          <>
            Sensing answers what the prescription is. Something still has to act on that answer. We plan
            to use an Alvarez lens: two cubic-profile plastic plates whose combined power changes when
            they slide sideways. Nothing on this page has been built yet. It is the design model that the
            lens bench will be measured against.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="why" num="3.1" title="Why not liquid crystal">
          <p>
            Electrically tunable lenses already exist, and almost all of them use liquid crystal. Two
            things make them a poor fit for all-day glasses. They cost on the order of $1,000 per lens at
            current volumes, and they need a voltage applied more or less continuously to hold a setting.
          </p>
          <p>
            The Alvarez lens was patented in 1967 and is now in the public domain. Its plates are
            injection-mouldable, the same process used for ordinary sunglasses lenses, so a rough estimate
            at production volume is single-digit dollars per lens. The optical power is set by plate
            position alone, so a mechanism that latches can hold a prescription with no power at all.
          </p>
        </Section>

        <Section id="plates" num="3.2" title="Two plates, one lens">
          <p>
            Each plate is flat on one side and carries a cubic surface on the other. The second plate
            carries the same surface with the opposite sign:
          </p>
          <Tex block>{String.raw`t(x, y) = A\left(x y^{2} + \tfrac{x^{3}}{3}\right)`}</Tex>
          <p>
            Stacked in register, the two thicknesses cancel and the pair is a flat window. Slide one plate
            by <Tex>+d</Tex> and the other by <Tex>-d</Tex> along <Tex>x</Tex>, and the cubic terms
            leave a quadratic difference:
          </p>
          <Tex block>{String.raw`t(x + d,\, y) - t(x - d,\, y) = 2Ad\,(x^{2} + y^{2}) + \tfrac{2}{3}Ad^{3}`}</Tex>
          <p>
            A thickness that grows as <Tex>{"x^{2} + y^{2}"}</Tex> is a thin spherical lens. The constant term
            is a uniform slab and has no optical effect. For plates of index <Tex>n</Tex> the power is
            linear in the displacement:
          </p>
          <Tex block>{String.raw`P = 4\,(n - 1)\,A\,d`}</Tex>
          <p>
            We design for a gain of {GAIN_D_PER_MM} D per millimetre in PMMA (<Tex>{`n = ${N_PMMA}`}</Tex>),
            which fixes <Tex>{String.raw`A \approx ${(A_PER_MM2 * 1000).toFixed(2)} \times 10^{-3}\ \text{mm}^{-2}`}</Tex>.
            The full ±6 D range then needs ±1.5 mm of travel per plate, and one 0.25 D clinical step is
            62.5 µm.
          </p>
          <Figure
            num={1}
            caption={
              <>
                A complementary plate pair and the ray bundle it produces. The cubic sag and the ray angles
                are exaggerated so that the geometry is visible. Switch to the astigmatic pair to slide in
                two directions.
              </>
            }
          >
            <AlvarezLensFigure />
          </Figure>
        </Section>

        <Section id="axes" num="3.3" title="Covering all three components">
          <p>
            A full correction needs all three <Link to="/simulation#power-vectors">power-vector</Link>{" "}
            components. A second pair with a different cubic supplies the two astigmatic ones:
          </p>
          <Tex block>{String.raw`t_{\text{astig}}(x, y) = A\left(\tfrac{x^{3}}{3} - x y^{2}\right)`}</Tex>
          <p>
            Sliding this pair along <Tex>x</Tex> gives a thickness change proportional to{" "}
            <Tex>{"x^{2} - y^{2}"}</Tex>, which is <Tex>J_0</Tex>. Sliding it along <Tex>y</Tex> gives a
            change proportional to <Tex>xy</Tex>, which is <Tex>{"J_{45}"}</Tex>. With the same coefficient{" "}
            <Tex>A</Tex>, all three axes share one gain.
          </p>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead>
                <tr><th>Component</th><th>Plate pair</th><th>Slide direction</th><th className={a.num}>Travel for full range</th></tr>
              </thead>
              <tbody>
                <tr><td><Tex>M</Tex></td><td>Spherical</td><td>x</td><td className={a.num}>±1.5 mm (±6 D)</td></tr>
                <tr><td><Tex>J_0</Tex></td><td>Astigmatic</td><td>x</td><td className={a.num}>±0.375 mm (±1.5 D)</td></tr>
                <tr><td><Tex>{"J_{45}"}</Tex></td><td>Astigmatic</td><td>y</td><td className={a.num}>±0.375 mm (±1.5 D)</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            The lens command is the estimate from PupilNet with its sign flipped. Because power vectors
            add exactly, the residual error of the eye plus the lens equals the estimation error, and
            each component maps to one actuator axis with no coupling between them.
          </p>
        </Section>

        <Section id="actuator" num="3.4" title="Moving the plates">
          <p>
            For the bench we model each axis as a plate on a flexure, pushed by a small voice coil and
            read by a Hall-effect position sensor. A 10 kHz controller combines a feedforward term with
            PID feedback. Figure 2 integrates this model live for any step in commanded power.
          </p>
          <Figure
            num={2}
            caption="Simulated step response of one plate axis. Position is in micrometres of plate travel. The second panel shows the coil current the controller commands, including the current needed to hold the plate against the flexure."
          >
            <ActuatorFigure />
          </Figure>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Model parameter</th><th className={a.num}>Value</th></tr></thead>
              <tbody>
                {ACT_ROWS.map(([k, v]) => (<tr key={k}><td>{k}</td><td className={a.num}>{v}</td></tr>))}
              </tbody>
            </table>
          </div>
          <Note label="Placeholder values.">
            <p>
              These actuator numbers are illustrative, chosen to be physically plausible for a small
              voice-coil stage. They are not measurements. The shape of the response is meaningful and
              the exact timings are not.
            </p>
          </Note>
          <p>
            The model also exposes a real design tension. A flexure pushes back, so this actuator draws
            about {(AMPS_PER_DIOPTRE * 1000).toFixed(0)} mA per dioptre just to stay in place. That is
            acceptable on a bench and not in a pair of glasses. The zero-power hold that makes the
            Alvarez lens attractive requires a mechanism that latches, such as a lead screw or a
            friction-held stage. Which one to use is an open question for the lens bench.
          </p>
        </Section>

        <Section id="measure" num="3.5" title="What the bench must measure">
          <p>
            The Alvarez lens will be built and characterised on its own bench before it meets the sensing
            side. The <Link to="/model-eye">model eye</Link> gives it a known error to correct.
          </p>
          <ol>
            <li>
              <strong>Gain.</strong> Does a measured slide produce the predicted change in power? This is
              the direct test of <Tex>{String.raw`P = 4(n-1)Ad`}</Tex> for a real moulded or machined
              plate.
            </li>
            <li>
              <strong>Zero and repeatability.</strong> Where is the true 0 D position, and does the lens
              return to it after a full-range move?
            </li>
            <li>
              <strong>Image quality off-axis.</strong> The thin-plate analysis ignores the air gap
              between plates and higher-order terms. Both grow with displacement and field angle.
            </li>
            <li>
              <strong>Speed and power.</strong> How long does a 0.25 D correction take, and what does the
              actuator draw while moving and while holding?
            </li>
          </ol>
          <p>
            Once those hold, both benches combine into the first complete physical loop: a real
            photorefraction camera, a network fine-tuned on real frames, and a real lens driven by its
            estimate.
          </p>
        </Section>
      </Article>
    </>
  );
}
