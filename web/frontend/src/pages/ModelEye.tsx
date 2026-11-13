import { Link } from "react-router-dom";
import { Article, Figure, Note, PageHeader, Section, Stat, StatRow, articleStyles as a, type TocEntry } from "../components/Article";

const TOC: TocEntry[] = [
  { id: "why", num: "2.1", title: "Why a model eye" },
  { id: "setup", num: "2.2", title: "Instrument and protocol" },
  { id: "accuracy", num: "2.3", title: "Accuracy" },
  { id: "images", num: "2.4", title: "Photographs against simulation" },
  { id: "band", num: "2.5", title: "The blind band" },
  { id: "limits", num: "2.6", title: "Limitations" },
];

const BLOCKS: [string, string, string, string, number][] = [
  ["A", "37 cm", "+4 D", "7 mm", 21],
  ["B", "37 cm", "+8 D", "7 mm", 11],
  ["C", "37 cm", "none", "7 mm", 37],
  ["D1", "37 cm", "none", "5 mm", 19],
  ["D2", "37 cm", "+5 D", "5 mm", 11],
  ["E1", "30 cm", "none", "7 mm", 19],
  ["E2", "30 cm", "+5 D", "7 mm", 11],
  ["F1", "25 cm", "none", "7 mm", 19],
  ["F2", "25 cm", "+6 D", "7 mm", 11],
  ["R", "37 cm", "+4 D", "7 mm", 11],
];

export function ModelEye() {
  return (
    <>
      <PageHeader
        kicker="Part 2: validation"
        title="A physical eye with a known prescription"
        lede={
          <>
            The network is trained entirely on simulated images. To measure how well it reads real
            photographs, the camera was pointed at a schematic eye whose refraction is set on a dial, at
            170 settings. At the design distance of 37 cm, with a moderate trial lens, the mean absolute
            error is 0.20 D.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="why" num="2.1" title="Why a model eye">
          <p>
            A human eye cannot supply ground truth: its clinical refraction is itself uncertain by
            0.25 to 0.5 D, and it moves, blinks and changes focus during a session. A schematic eye has a
            fixed, adjustable refraction, does not accommodate, and stays where it is placed. Errors
            measured on it belong to the camera, the image analysis and the network, which is the
            quantity under test.
          </p>
        </Section>

        <Section id="setup" num="2.2" title="Instrument and protocol">
          <p>
            The eye is an XSJ-501 schematic eye: refraction from +4 to &minus;5 D in 0.25 D steps, with
            3, 5 and 7 mm pupils. Trial lenses of +4, +5, +6 and +8 D in front of it extend the range to
            &minus;13 D. The camera is a Raspberry Pi 4 with an IMX708 NoIR sensor; two 850 nm LEDs sit
            10 mm from the lens, one to the right (H) and one above (V), switched by an Arduino between
            frames. Every shot consists of an H frame, a V frame and a dark frame, recorded as raw
            10-bit data.
          </p>
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Block</th><th>Distance</th><th>Trial lens</th><th>Pupil</th><th>Settings</th></tr></thead>
              <tbody>
                {BLOCKS.map(([b, d, l, p, n]) => (
                  <tr key={b}><td>{b}</td><td>{d}</td><td>{l}</td><td>{p}</td><td>{n}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Each setting was photographed 20 times: 3,400 shots and 10,200 frames in total. Block R
            repeats block A after the eye was removed from the bench and placed again, to measure
            repeatability.
          </p>
        </Section>

        <Section id="accuracy" num="2.3" title="Accuracy">
          <Figure
            num={1}
            caption={
              <>
                Network reading against the refraction set on the dial, for the six trial-lens blocks.
                Filled points: a crescent is visible. Open points: the eye lies in the blind band and the
                photograph carries no refraction signal. The shaded band marks &plusmn;0.5 D.
              </>
            }
          >
            <img src="/figures/model_eye_fig1_accuracy.png" alt="Network reading against the dial setting for blocks A, D2, R, B, E2 and F2" style={{ width: "100%", display: "block", background: "#fff" }} />
          </Figure>
          <StatRow>
            <Stat value="0.20" unit="D" label="mean absolute error, blocks A, D2 and R (26 settings)" />
            <Stat value="0.22" unit="D" label="largest A-to-R difference at 6 of 7 repeated settings" />
            <Stat value="0.49" unit="D" label="mean absolute error, all six trial-lens blocks (47 settings)" />
            <Stat value="0.90" unit="D" label="mean absolute error with the +8 D lens (block B)" />
          </StatRow>
          <p>
            Accuracy is highest at 37 cm with a +4 or +5 D trial lens. The +8 D lens and the shorter
            distances read too weak, by 0.6 to 1.0 D on average. Section 5.5 of the{" "}
            <Link to="/network">network page</Link> describes what has been ruled out as the cause.
          </p>
        </Section>

        <Section id="images" num="2.4" title="Photographs against simulation">
          <Figure
            num={2}
            caption={
              <>
                Photographs of the model eye (blue frames) beside simulated frames at the same refraction
                (orange frames), shown with the crop and brightness normalisation the network receives.
                The side of the crescent and its growth with refraction agree. Photographed crescents are
                softer and shorter in the vertical direction than simulated ones, which is the main
                remaining difference between the two.
              </>
            }
          >
            <img src="/figures/model_eye_fig2_real_vs_simulated.png" alt="Photographed and simulated pupil frames for H and V LEDs at six refractions" style={{ width: "100%", maxWidth: 560, display: "block", margin: "0 auto", background: "#fff" }} />
          </Figure>
        </Section>

        <Section id="band" num="2.5" title="The blind band">
          <Figure
            num={3}
            caption={
              <>
                (a) H-LED frames from block A. (b) Position of the reflected light within the pupil
                against the refraction of eye and trial lens, at 37 cm with a 7 mm pupil, for blocks A,
                B and C, with the simulator's prediction. The measured band edges lie 0.3 to 0.8 D
                outside the simulated ones.
              </>
            }
          >
            <img src="/figures/model_eye_fig3_physics.png" alt="Crescent frames and crescent position against refraction, measured and simulated" style={{ width: "100%", display: "block", background: "#fff" }} />
          </Figure>
          <p>
            Near the refraction at which the eye is focused on the camera, no light returns into the
            camera aperture and the pupil appears dark. At 37 cm with a 7 mm pupil this blind band
            extends from about &minus;5.7 to +1.9 D of combined eye and lens power. Of the 170 settings,
            107 lie inside it. A trial lens moves the eye out of the band; in the glasses design, the
            Alvarez lens performs the same function.
          </p>
        </Section>

        <Section id="limits" num="2.6" title="Limitations">
          <Note label="Scope.">
            <p>
              A schematic eye has no accommodation, no tear film and no eyelids, and its retina reflects
              differently from a human retina. The model-eye results establish that the
              simulator-trained network can read real optics; validation on human eyes is a separate
              step.
            </p>
          </Note>
        </Section>
      </Article>
    </>
  );
}
