import { Link } from "react-router-dom";
import {
  Article, Note, PageHeader, Section, Stat, StatRow, Tex, articleStyles as a, type TocEntry,
} from "../components/Article";

const TOC: TocEntry[] = [
  { id: "what", num: "5.1", title: "What the network is" },
  { id: "bench", num: "5.2", title: "What it measured on a real eye" },
  { id: "troubles", num: "5.3", title: "Troubles, and what fixed them" },
  { id: "failed", num: "5.4", title: "Fixes that failed the test" },
  { id: "trust", num: "5.5", title: "Knowing when a reading is wrong" },
];

export function Network() {
  return (
    <>
      <PageHeader
        kicker="5 · Network"
        title="PupilNet: the estimator, and everything that went wrong"
        lede={
          <>
            A 152,451-parameter convolutional network reads two infrared pupil frames and returns the
            refraction. It has never seen a real photograph during training &mdash; every training image
            comes from our ray tracer &mdash; yet it reads a physical model eye to 0.20 D at the design
            distance. This page explains how, and documents each failure on the way, because the failures
            are the useful part.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="what" num="5.1" title="What the network is">
          <p>
            Input: the two 64&times;64 LED frames, cropped so the pupil fills a fixed fraction of the
            image, plus four numbers the photo cannot reveal &mdash; camera distance, LED offset, pupil
            radius, camera aperture. Output: the spherical equivalent <Tex>M</Tex> (the deployed
            checkpoint is trained for <Tex>M</Tex> only; astigmatism stays in the photos as a nuisance).
            The architecture is a small depthwise-separable CNN: a stem, four halving blocks, average
            pooling, two dense layers. It is deliberately tiny &mdash; about 150&nbsp;KB at 8-bit &mdash;
            because it has to run on the <Link to="/prototype">module's microcontroller</Link>.
          </p>
          <p>
            The two frames are normalised by one shared brightness scale so exposure cancels but the
            ratio between the LEDs survives. Training data: 100k simulated frames spanning
            &minus;12&hellip;+6 D, 25&ndash;100 cm, randomized eyes (pupil, aberration, scatter), sensors
            (noise, blur, exposure) and scenes (iris texture, glints, eyelids, lashes).
          </p>
          <StatRow>
            <Stat value="152,451" label="parameters; ~150 KB quantized" />
            <Stat value="0" label="real photographs in training" />
            <Stat value="0.65" unit="D" label="mean |error| on held-out simulated frames with a visible crescent" />
            <Stat value="0.20" unit="D" label="mean |error| on the physical model eye, 37 cm, +4/+5 D lens (26 settings)" />
          </StatRow>
        </Section>

        <Section id="bench" num="5.2" title="What it measured on a real eye">
          <p>
            In November 2026 we photographed an XSJ-501 schematic eye &mdash; a metal eye whose refraction
            is set on a dial &mdash; at 170 settings (3,400 shots, each an H, V and dark frame), across
            three distances, two pupils and four trial lenses. Full details are on the{" "}
            <Link to="/model-eye">model-eye page</Link>; the headline rows:
          </p>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Block</th><th>Setup</th><th>Mean |error|</th><th>Within 0.5 D</th></tr></thead>
              <tbody>
                <tr><td>A</td><td>37 cm, +4 D lens, 7 mm pupil</td><td>0.23 D</td><td>100%</td></tr>
                <tr><td>D2</td><td>37 cm, +5 D lens, 5 mm pupil</td><td>0.07 D</td><td>100%</td></tr>
                <tr><td>R</td><td>A again after re-placing the eye</td><td>0.26 D</td><td>86%</td></tr>
                <tr><td>B</td><td>37 cm, +8 D lens</td><td>0.90 D</td><td>18%</td></tr>
                <tr><td>E2</td><td>30 cm, +5 D lens</td><td>0.66 D</td><td>33%</td></tr>
                <tr><td>F2</td><td>25 cm, +6 D lens</td><td>1.01 D</td><td>0%</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            Repeatability after taking the eye off the table and putting it back: six of seven settings
            agreed within 0.22 D. Two caveats belong next to the good rows: only 63 of the 170 settings
            show a crescent at all (the rest sit in the blind band, which is physics, and is what the
            trial lens is for), and the strong-lens and close-distance rows are not solved.
          </p>
        </Section>

        <Section id="troubles" num="5.3" title="Troubles, and what fixed them">
          <p>Every error above 0.5 D we chased had a physical cause. In the order we hit them:</p>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Trouble</th><th>Symptom</th><th>Fix, and the evidence</th></tr></thead>
              <tbody>
                <tr>
                  <td>The dark circle in the photo is not the pupil</td>
                  <td>Readings shifted ~4 D</td>
                  <td>The model eye's front lens magnifies its aperture ~1.8&times;. Use the set 5/7 mm size; the light returning through the aperture fills a disc of exactly that size.</td>
                </tr>
                <tr>
                  <td>The aperture is off-centre in that circle</td>
                  <td>One block read nonsense (11.8 D mean error)</td>
                  <td>Find the aperture from where crescents appear across a whole block, pooling lens and no-lens blocks so both sides are lit. That block went from 11.8 to 0.07 D.</td>
                </tr>
                <tr>
                  <td>Pupil centring sensitivity</td>
                  <td>0.5 mm off costs ~0.27 D (median), up to 0.67 D</td>
                  <td>Measured by shifting the centre on purpose. Centring is now treated as a first-class calibration, not a preprocessing detail.</td>
                </tr>
                <tr>
                  <td>Reflections fooled the eye-finder</td>
                  <td>Crops locked onto trial-lens glints and bright cloth</td>
                  <td>Mask clipped pixels, search near the centre, lock the crop per block.</td>
                </tr>
                <tr>
                  <td>Sensor black level read as room light</td>
                  <td>A dark room flagged &ldquo;too bright&rdquo;</td>
                  <td>Subtract the sensor's fixed offset before judging.</td>
                </tr>
                <tr>
                  <td>Training range ended at &minus;12 D</td>
                  <td>Readings flattened beyond it</td>
                  <td>Retrained &minus;16&hellip;+8 D: the flattening went away (slope 0.83 &rarr; 1.11 on the +8 D block) but mid-range accuracy got 0.4 D worse, so the original stays deployed. An honest trade, documented, not hidden.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="failed" num="5.4" title="Fixes that failed the test">
          <p>
            Two plausible explanations were tested against the photos and made things <em>worse</em>;
            both are recorded so nobody retries them by accident.
          </p>
          <ul>
            <li>
              <strong>&ldquo;The LED must really be at 11.5 mm.&rdquo;</strong> The blind-band edges fitted
              11.5 mm better than the built 10 mm. Telling the network 11.5 mm degraded every block
              (0.51 &rarr; 0.63 D mean). The band edges are noise-limited; the hardware was right.
            </li>
            <li>
              <strong>&ldquo;Train with realistic blur.&rdquo;</strong> Real crescents are softer than
              simulated ones (blur &asymp; 15% of pupil radius). Networks trained with that blur read the
              real photos 1.2&ndash;1.4 D too weak. Reverted.
            </li>
          </ul>
          <Note label="Still open.">
            <p>
              At the same total power (&minus;8.3 D), the +4 D lens block reads correctly and the +8 D lens
              block reads ~1 D weak. The eye, centring and analysis are identical, so the strong lens itself
              is doing something our model misses. This is the main unsolved accuracy problem.
            </p>
          </Note>
        </Section>

        <Section id="trust" num="5.5" title="Knowing when a reading is wrong">
          <p>
            A measuring device must flag its own bad readings. We tested the standard ML answer and a
            physical one:
          </p>
          <ul>
            <li>
              <strong>Ensemble disagreement fails.</strong> Three independently trained copies agree with
              each other even when all three are wrong (rank correlation between disagreement and error:
              0.09, p = 0.49). They share the simulator's blind spots, so the network cannot grade itself.
            </li>
            <li>
              <strong>A lens step works.</strong> Change the lens by a known 1 D: the reading must move by
              1 D. This self-check needs no ground truth, flags all three weak blocks above, and costs one
              extra photo. It is now part of the analysis, and the <Link to="/prototype">prototype's</Link>{" "}
              sweep gets it for free.
            </li>
          </ul>
        </Section>
      </Article>
    </>
  );
}
