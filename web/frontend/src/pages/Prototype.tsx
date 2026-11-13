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
  frame: "Carries both Alvarez lens stacks. No electronics: the glasses stay cheap and dumb.",
  plates: "Two moulded cubic plates per eye; sliding them sideways sets sphere power (Alvarez, 1967). Ours came from consumer adjustable glasses with the knob removed.",
  slider: "The tab the module's motor grips, so the module can set the lens power it is measuring through.",
  temples: "Standard temples; the module clamps to the frame, not to the head.",
  housing: "A light-tight tube. It also blocks this eye's view, which matters: a covered eye relaxes with its open partner (consensual accommodation).",
  relay: "The key optic. With the camera at its focal plane, the camera behaves as if it were at optical infinity, so “no crescent” means exactly “eye + lens corrected”.",
  camera: "A small NoIR sensor at the relay focus, 100 mm from the relay. Our bench uses a Raspberry Pi IMX708; the wearable wants a board camera.",
  leds: "Two surface-mount 850 nm LEDs, about 2 mm off the camera axis. Our simulation says 1 mm is too close: the pupil never goes dark, so there is no band to find.",
  motor: "A micro lead-screw stepper that slides the plates. Accuracy comes from the camera loop, not from the motor: the loop only needs the sign and rough size of the step.",
  encoder: "Reads the plate position so the converged setting can be reported in dioptres after a one-time lensmeter calibration.",
  mcu: "Runs the 152k-parameter network (about 150 KB at 8-bit). Candidate: a camera-capable microcontroller board.",
  battery: "Sized for sessions, not all-day wear: a measurement is seconds long.",
};

const BOM: { part: string; status: "have" | "to buy" | "to decide"; note: string }[] = [
  { part: "Alvarez glasses (plates + slider)", status: "have", note: "consumer adjustable pair, knob removed; characterize power-per-mm on a lensmeter" },
  { part: "IMX708 NoIR camera + Raspberry Pi 4", status: "have", note: "bench stand-in for the module camera" },
  { part: "850 nm LEDs (5 mm, through-hole)", status: "have", note: "bench only; module needs surface-mount parts at 2 mm offset" },
  { part: "Arduino LED driver", status: "have", note: "switches H/V LEDs between frames" },
  { part: "XSJ-501 model eye + trial lenses", status: "have", note: "ground truth for every bench test" },
  { part: "Relay lens, f = 100 mm, Ø25 mm achromat", status: "to buy", note: "sets module length; coated for near-infrared" },
  { part: "Micro lead-screw stepper + driver", status: "to buy", note: "20–30 µm step if plate gain is 4–6 D/mm — but the loop relaxes this" },
  { part: "Position encoder (linear strip)", status: "to buy", note: "needed only to report the number, not to converge" },
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
        title="The instrument we are building"
        lede={
          <>
            Cheap adjustable glasses, and a clip-on module that measures the eye through its own lens and
            then sets that lens. The model below is the current design intent, drawn to scale in
            millimetres. Drag to orbit, click a part to read about it, and pull the assembly apart.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="model" num="4.1" title="The prototype, part by part">
          <Figure
            num={1}
            caption={
              <>
                Design model of the prototype. The tube clips in front of the right lens; its length is the
                relay's 100 mm focal distance. Dashed blue: the measurement path from pupil through the
                Alvarez stack and relay to the camera. Dimensions are design intent, not a measured drawing.
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
          <Panel title={selected ? PART_LABELS[selected] : "Click a part"}>
            <p style={{ margin: 0 }}>
              {selected ? PART_NOTES[selected] : "Each part in the model is clickable. The numbers match the bill of parts below."}
            </p>
          </Panel>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>#</th><th>Part</th><th>Why it is there</th></tr></thead>
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

        <Section id="measure" num="4.2" title="How a measurement runs">
          <ol>
            <li>Clip the module over one lens. That eye sees only darkness; the other eye looks at something far away, and both eyes relax together (accommodation is consensual).</li>
            <li>The LEDs flash, the camera takes pupil frames through the Alvarez lens and the relay.</li>
            <li>The motor sweeps the lens until the crescent vanishes and reappears on the other side; the midpoint of that dark band is the null. Because the camera sits at the relay focus, the null means <em>eye + lens focused at infinity</em> &mdash; the lens is now the prescription.</li>
            <li>The encoder reports the converged plate position in dioptres. Then the module moves to the other lens and repeats.</li>
          </ol>
          <p>
            The loop never needs to trust the motor. A sticky, imprecise actuator still converges, because
            every frame re-measures the residual error through the very lens being set. What the sweep{" "}
            <em>cannot</em> fix is an eye that re-focuses mid-measurement; that is the{" "}
            <Link to="/control">control problem</Link>, and covering the measured eye is half of the answer.
          </p>
          <Note label="Checked in simulation.">
            <p>
              With the LED 2 mm off axis and a 5 mm pupil, the dark band spans about <Tex>{"-2.25"}</Tex> to{" "}
              <Tex>{"+2.0"}</Tex> D around the null and its midpoint lands within 0.125 D of zero for 3&ndash;7 mm
              pupils &mdash; but only for an aberration-free eye. Real spherical aberration shifts the midpoint
              by up to about 1 D at large pupils, which is why the band edges should be read at a small pupil,
              or corrected with a second measurement (see <Link to="/network">the network page</Link> for how
              pupil geometry already bit us once).
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
            <li><strong>Alvarez plates in the measurement path.</strong> The camera looks through them: their off-centre astigmatism, 850 nm transmission and surface glints must be measured before the module is built around them.</li>
            <li><strong>Range.</strong> Reading both band edges needs the lens to reach prescription &plusmn; about 2.5 D. A &minus;6.75 D eye with a &minus;6&hellip;+3 D lens cannot; there the network reads the crescent on one readable side instead of bracketing.</li>
            <li><strong>Vertex distance and tilt.</strong> The module must sit repeatably; our bench showed 0.5 mm of pupil mis-centring already costs ~0.27 D.</li>
            <li><strong>Bright rooms.</strong> Small pupils widen the band. The midpoint method tolerates that; one-sided network readings get harder.</li>
            <li><strong>Eye safety.</strong> 850 nm exposure at a few centimetres must be checked against the exposure limits before any human use.</li>
          </ul>
        </Section>
      </Article>
    </>
  );
}
