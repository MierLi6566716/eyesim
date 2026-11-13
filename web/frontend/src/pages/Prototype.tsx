import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Article, Figure, Note, PageHeader, Panel, Section, Tex, articleStyles as a, type TocEntry,
} from "../components/Article";
import { GlassesScene, PART_LABELS, type PartId } from "../three/GlassesScene";

const TOC: TocEntry[] = [
  { id: "model", num: "4.1", title: "The prototype, part by part" },
  { id: "measure", num: "4.2", title: "How a measurement runs" },
  { id: "parts", num: "4.3", title: "Bill of parts and status" },
  { id: "open", num: "4.4", title: "Open engineering questions" },
];

const PART_NOTES: Record<PartId, string> = {
  frame: "Carries both Alvarez lens stacks. The glasses contain no electronics.",
  plates: "Two moulded cubic plates per eye; sliding them sideways sets the spherical power (Alvarez, 1967). The current plates come from consumer adjustable glasses with the adjustment knob removed.",
  slider: "The tab engaged by the module's motor, which allows the module to set the lens through which it measures.",
  temples: "Standard temples. The module clamps to the frame.",
  housing: "A light-tight tube. It covers the measured eye, whose accommodation then follows the uncovered eye (consensual accommodation).",
  relay: "With the camera at its focal plane, the camera is optically at infinity, so the absence of a crescent indicates that eye and lens together are corrected.",
  camera: "A small near-infrared sensor at the relay focus, 100 mm from the relay. The bench uses a Raspberry Pi IMX708; the wearable requires a board camera.",
  leds: "Two surface-mount 850 nm LEDs, about 2 mm from the camera axis. In simulation an offset of 1 mm is too small: the pupil never becomes dark, and there is no band to locate.",
  motor: "A micro lead-screw stepper that slides the plates. Accuracy is set by the camera loop rather than by the motor, which needs only the correct direction and an approximate step size.",
  encoder: "Reads the plate position, so that the final setting can be reported in dioptres after a one-time lensmeter calibration.",
  mcu: "Runs the 152,451-parameter network (about 150 KB at 8-bit precision) on a camera-capable microcontroller.",
  battery: "Sized for occasional sessions rather than continuous wear; a measurement lasts seconds.",
};

const BOM: { part: string; status: "have" | "to buy" | "to decide"; note: string }[] = [
  { part: "Alvarez glasses (plates + slider)", status: "have", note: "consumer adjustable pair, knob removed; characterize power-per-mm on a lensmeter" },
  { part: "IMX708 NoIR camera + Raspberry Pi 4", status: "have", note: "bench stand-in for the module camera" },
  { part: "850 nm LEDs (5 mm, through-hole)", status: "have", note: "bench only; module needs surface-mount parts at 2 mm offset" },
  { part: "Arduino LED driver", status: "have", note: "switches H/V LEDs between frames" },
  { part: "XSJ-501 model eye + trial lenses", status: "have", note: "ground truth for every bench test" },
  { part: "Relay lens, f = 100 mm, Ø25 mm achromat", status: "to buy", note: "sets module length; coated for near-infrared" },
  { part: "Micro lead-screw stepper + driver", status: "to buy", note: "20–30 µm step for a plate gain of 4–6 D/mm; the closed loop relaxes this requirement" },
  { part: "Position encoder (linear strip)", status: "to buy", note: "required to report the setting, not for convergence" },
  { part: "Module controller (camera-capable MCU)", status: "to decide", note: "depends on 8-bit network latency per frame" },
  { part: "Housing (3D-printed tube + clamp)", status: "to decide", note: "first version can be printed around the bench camera" },
];

export function Prototype() {
  const [spread, setSpread] = useState(0.65);
  const [selected, setSelected] = useState<PartId | null>(null);
  const [rays, setRays] = useState(true);

  return (
    <>
      <PageHeader
        kicker="4 · Prototype"
        title="The measuring instrument"
        lede={
          <>
            The instrument consists of inexpensive adjustable glasses and a clip-on module that measures
            the eye through the glasses' own lens and then sets that lens. The model below shows the
            current design to scale in millimetres; each part can be selected, and the assembly can be
            separated into its components.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="model" num="4.1" title="Components">
          <Figure
            num={1}
            caption={
              <>
                Design model of the prototype. The module attaches in front of the right lens; its length
                is set by the 100 mm focal length of the relay lens. Dashed line: measurement path from the
                pupil through the Alvarez plates and the relay lens to the camera. Dimensions are nominal.
              </>
            }
          >
            <div style={{ height: 430, border: "1px solid var(--rule)", background: "#fff" }}>
              <GlassesScene spread={spread} selected={selected} onPick={setSelected} showRays={rays} />
            </div>
            <div style={{ display: "flex", gap: 18, alignItems: "center", marginTop: 10, flexWrap: "wrap" }}>
              <label className="label-caps" style={{ display: "flex", gap: 10, alignItems: "center" }}>
                assembled
                <input
                  type="range" min={0} max={1} step={0.01} value={spread}
                  onChange={(e) => setSpread(Number(e.target.value))}
                  style={{ width: 180 }}
                  aria-label="explode the assembly"
                />
                exploded
              </label>
              <label className="label-caps" style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <input type="checkbox" checked={rays} onChange={(e) => setRays(e.target.checked)} />
                measurement path
              </label>
            </div>
          </Figure>
          <Panel title={selected ? PART_LABELS[selected] : "Selected part"}>
            <p style={{ margin: 0 }}>
              {selected ? PART_NOTES[selected] : "None selected. Part numbers correspond to the table below."}
            </p>
          </Panel>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>#</th><th>Part</th><th>Function</th></tr></thead>
              <tbody>
                {(Object.keys(PART_LABELS) as PartId[]).map((id) => (
                  <tr key={id} style={selected === id ? { background: "var(--blue-soft)" } : undefined}>
                    <td>{PART_LABELS[id].split(" · ")[0]}</td>
                    <td><button onClick={() => setSelected(id)} style={{ border: "none", padding: 0, background: "none", textDecoration: "underline", cursor: "pointer", font: "inherit" }}>{PART_LABELS[id].split(" · ")[1]}</button></td>
                    <td>{PART_NOTES[id]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="measure" num="4.2" title="Measurement sequence">
          <ol>
            <li>The module is attached over one lens. The measured eye sees only the dark interior of the module, while the other eye views a distant target; because accommodation is consensual, both eyes relax together.</li>
            <li>The LEDs are pulsed and the camera records pupil frames through the Alvarez lens and the relay lens.</li>
            <li>The motor sweeps the lens until the crescent disappears and reappears on the opposite side. Because the camera is optically at infinity, the centre of this dark band corresponds to eye and lens together focused at infinity, so the lens setting equals the prescription.</li>
            <li>The encoder reports the final plate position in dioptres, and the procedure is repeated on the other eye.</li>
          </ol>
          <p>
            The loop does not depend on the accuracy of the motor: each frame measures the remaining error
            through the lens being set, so an imprecise actuator still converges. The sweep does not
            compensate for an eye that changes focus during the measurement. That problem is treated on
            the <Link to="/control">control page</Link>; covering the measured eye removes much of it.
          </p>
          <Note label="Simulation check.">
            <p>
              With the LED 2 mm from the camera axis and a 5 mm pupil, the dark band extends from about{" "}
              <Tex>{"-2.25"}</Tex> to <Tex>{"+2.0"}</Tex> D around the null. For an eye without aberrations
              its centre lies within 0.125 D of zero for pupils of 3 to 7 mm. Spherical aberration shifts
              the centre by up to about 1 D at large pupils; the band is therefore read at a small pupil, or
              at two pupil sizes from which the shift is extrapolated.
            </p>
          </Note>
        </Section>

        <Section id="parts" num="4.3" title="Bill of parts and status">
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Part</th><th>Status</th><th>Notes</th></tr></thead>
              <tbody>
                {BOM.map((r) => (
                  <tr key={r.part}>
                    <td>{r.part}</td>
                    <td style={{ whiteSpace: "nowrap", color: r.status === "have" ? "var(--green)" : r.status === "to buy" ? "var(--blue)" : "var(--ink-3)" }}>{r.status}</td>
                    <td>{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="open" num="4.4" title="Open engineering questions">
          <ul>
            <li><strong>Alvarez plates in the measurement path.</strong> The camera views the eye through the plates, so their off-centre astigmatism, transmission at 850 nm and surface reflections constrain the module design.</li>
            <li><strong>Range.</strong> Locating both band edges requires the lens to reach the prescription &plusmn; about 2.5 D. Where the plates cannot reach that range, the network reads the crescent on one side of the band instead.</li>
            <li><strong>Vertex distance and tilt.</strong> The module must be positioned repeatably; on the bench, a 0.5 mm pupil-centring error changed the reading by 0.27 D (median).</li>
            <li><strong>Illumination.</strong> Small pupils widen the band. The midpoint method is unaffected; single-sided network readings become more difficult.</li>
            <li><strong>Eye safety.</strong> Infrared exposure at a distance of a few centimetres requires verification against exposure limits before use on human eyes.</li>
          </ul>
        </Section>
      </Article>
    </>
  );
}
