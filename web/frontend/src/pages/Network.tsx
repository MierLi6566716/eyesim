import { Link } from "react-router-dom";
import {
  Article, Note, PageHeader, Section, Stat, StatRow, Tex, articleStyles as a, type TocEntry,
} from "../components/Article";

const TOC: TocEntry[] = [
  { id: "what", num: "5.1", title: "Architecture and training" },
  { id: "bench", num: "5.2", title: "Results on the model eye" },
  { id: "human", num: "5.3", title: "Results on a human eye" },
  { id: "problems", num: "5.4", title: "Problems and corrections" },
  { id: "rejected", num: "5.5", title: "Hypotheses rejected" },
  { id: "trust", num: "5.6", title: "Detecting unreliable readings" },
];

export function Network() {
  return (
    <>
      <PageHeader
        kicker="5 · Network"
        title="PupilNet: the refraction estimator"
        lede={
          <>
            A convolutional network of 152,451 parameters estimates refraction from two infrared pupil
            frames. It is trained only on simulated images and reaches a mean absolute error of 0.20 D on
            a physical model eye at its design distance. On a human eye it does not yet perform at that
            level. This page describes the network, the results, and the errors identified so far.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="what" num="5.1" title="Architecture and training">
          <p>
            The input is a pair of 64&times;64 frames, one per LED, cropped so that the pupil fills a
            fixed fraction of the image, together with four quantities that a photograph does not
            reveal: camera distance, LED offset, pupil radius and camera aperture. The output is the
            spherical equivalent <Tex>M</Tex>; astigmatism remains in the images as a nuisance variable.
            The network consists of a convolutional stem, four depthwise-separable blocks that each halve
            the resolution, average pooling and two dense layers. At 8-bit precision it occupies about
            150 KB, small enough for a microcontroller.
          </p>
          <p>
            Both frames are divided by a common brightness scale, which removes exposure while preserving
            the intensity ratio between the two LEDs. The base network was trained on 300,000 simulated
            frames for 30 epochs; the network used for all measurements here was then fine-tuned on
            100,000 frames covering &minus;12 to +6 D and 25 to 100 cm, with randomised pupil size,
            aberration, retinal scatter, sensor noise, blur, glints, eyelids and lashes.
          </p>
          <StatRow>
            <Stat value="152,451" label="parameters" />
            <Stat value="0" label="real photographs used in training" />
            <Stat value="0.65" unit="D" label="mean absolute error, held-out simulated frames with a visible crescent" />
            <Stat value="0.20" unit="D" label="mean absolute error, model eye at 37 cm, +4 or +5 D lens (26 settings)" />
          </StatRow>
        </Section>

        <Section id="bench" num="5.2" title="Results on the model eye">
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Block</th><th>Configuration</th><th>Mean absolute error</th><th>Within 0.5 D</th></tr></thead>
              <tbody>
                <tr><td>A</td><td>37 cm, +4 D lens, 7 mm pupil</td><td>0.23 D</td><td>100%</td></tr>
                <tr><td>D2</td><td>37 cm, +5 D lens, 5 mm pupil</td><td>0.07 D</td><td>100%</td></tr>
                <tr><td>R</td><td>block A repeated after re-placing the eye</td><td>0.26 D</td><td>86%</td></tr>
                <tr><td>B</td><td>37 cm, +8 D lens</td><td>0.90 D</td><td>18%</td></tr>
                <tr><td>E2</td><td>30 cm, +5 D lens</td><td>0.66 D</td><td>33%</td></tr>
                <tr><td>F2</td><td>25 cm, +6 D lens</td><td>1.01 D</td><td>0%</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            Only settings whose photograph shows a crescent are scored: 63 of 170. The remainder lie in
            the blind band, where the image carries no refraction signal. Protocol and figures are on the{" "}
            <Link to="/model-eye">model-eye page</Link>.
          </p>
        </Section>

        <Section id="human" num="5.3" title="Results on a human eye">
          <p>
            The same camera photographed a myopic adult eye (spectacle prescription &minus;6.75 D, about
            &minus;6.2 D at the cornea) through trial lenses of +0.5 to +4 D. In every session the
            crescent appeared on the LED side, consistent with strong myopia, but the network read the
            eye as &minus;0.6 to &minus;5.2 D: an under-reading of 1 to 5.6 D that grows with the power
            of the trial lens. The readings saturate at a combined eye-and-lens power of about &minus;5 D,
            where the model eye was read correctly to &minus;9 D.
          </p>
          <p>
            A side-by-side comparison at equal combined power identifies the main difference. On the
            human eye the H-LED crescent is barely visible against a bright iris background, while the
            V-LED crescent is strong; on the model eye and in simulation both crescents are strong. The
            network therefore receives one usable crescent instead of two. Contributing factors are an
            imbalance between the two LEDs in brightness or aim, a dilated pupil of about 8 mm, the high
            infrared reflectance of the iris, and eye movement of up to 20 pixels during a session. These
            are capture conditions that can be corrected before the human measurement is repeated.
          </p>
        </Section>

        <Section id="problems" num="5.4" title="Problems and corrections">
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Problem</th><th>Effect</th><th>Correction and evidence</th></tr></thead>
              <tbody>
                <tr>
                  <td>The dark disc in the photograph is not the model eye's pupil</td>
                  <td>Readings shifted by about 4 D</td>
                  <td>The model eye's front lens magnifies its aperture by about 1.8. The set pupil size is used; light returning through the aperture fills a disc of exactly that size.</td>
                </tr>
                <tr>
                  <td>The aperture is off-centre within the dark disc</td>
                  <td>Block D2 mean error of 11.8 D</td>
                  <td>The aperture is located from where crescents appear across a block, pooling blocks with and without a trial lens so that both sides are lit. Block D2 error fell to 0.07 D.</td>
                </tr>
                <tr>
                  <td>Sensitivity to pupil centring</td>
                  <td>0.27 D median change per 0.5 mm (maximum 0.67 D)</td>
                  <td>Measured by displacing the centre by known amounts. Centring is treated as a calibration quantity.</td>
                </tr>
                <tr>
                  <td>Reflections misdirected the eye finder</td>
                  <td>Crops centred on trial-lens glints</td>
                  <td>Saturated pixels are masked, the search is restricted to the frame centre, and the crop is fixed per block.</td>
                </tr>
                <tr>
                  <td>Sensor black level interpreted as room light</td>
                  <td>Dark-room check failed</td>
                  <td>The sensor's fixed offset is subtracted before the check.</td>
                </tr>
                <tr>
                  <td>Training range limited to &minus;12 D</td>
                  <td>Readings flattened beyond it</td>
                  <td>A network trained over &minus;16 to +8 D removed the flattening (block B slope 0.83 to 1.11) but lost 0.4 D at moderate powers; the original network remains in use.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="rejected" num="5.5" title="Hypotheses rejected">
          <ul>
            <li>
              <strong>LED offset of 11.5 mm rather than the built 10 mm.</strong> The blind-band edges
              fitted 11.5 mm more closely. Supplying 11.5 mm to the network increased the mean error from
              0.51 to 0.63 D. The band edges are limited by noise, and the built value is correct.
            </li>
            <li>
              <strong>Camera blur.</strong> Photographed crescents are softer than simulated ones
              (blur of about 15% of the pupil radius). Networks trained with that blur read the
              photographs 1.2 to 1.4 D too weak, and the change was reverted.
            </li>
            <li>
              <strong>Pupil magnification by the trial lens.</strong> Correcting for it increased the
              error in every block.
            </li>
          </ul>
          <Note label="Unresolved.">
            <p>
              At an identical combined power of &minus;8.3 D, block A (+4 D lens) reads correctly and
              block B (+8 D lens) reads about 1 D too weak. Eye position, centring and analysis are the
              same, so the stronger trial lens introduces an effect that the model does not contain.
            </p>
          </Note>
        </Section>

        <Section id="trust" num="5.6" title="Detecting unreliable readings">
          <p>
            Two methods were tested for flagging readings that should not be trusted.
          </p>
          <ul>
            <li>
              <strong>Ensemble disagreement.</strong> Three independently trained copies of the network
              agree with one another even when all three are wrong: the rank correlation between their
              disagreement and the actual error is 0.09 (p = 0.49). They share the simulator's
              systematic errors and cannot detect them.
            </li>
            <li>
              <strong>Lens-step consistency.</strong> A known change of lens power must change the
              reading by the same amount. This check requires no ground truth and identifies all three
              weak blocks (B, E2, F2), with one false alarm (R). It is part of the analysis pipeline, and
              the lens sweep of the <Link to="/prototype">prototype</Link> provides it at no extra cost.
            </li>
          </ul>
        </Section>
      </Article>
    </>
  );
}
