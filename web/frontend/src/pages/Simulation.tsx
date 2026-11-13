import { useState } from "react";
import { Link } from "react-router-dom";
import { Article, Figure, Note, PageHeader, Section, Stat, StatRow, Tex, type TocEntry } from "../components/Article";
import type { Eye } from "../figures/EyeControls";
import { PowerVectorFigure } from "../figures/PowerVectorFigure";
import { OpticsChainFigure, PredistortFigure } from "../figures/OpticsFigures";
import { PhotorefractionFigure } from "../figures/PhotorefractionFigure";
import { TrainingFigure } from "../figures/TrainingFigure";
import { ClosedLoopFigure } from "../figures/ClosedLoopFigure";

const TOC: TocEntry[] = [
  { id: "power-vectors", num: "1.1", title: "Prescriptions as vectors" },
  { id: "wave-optics", num: "1.2", title: "Wavefront to retinal image" },
  { id: "predistortion", num: "1.3", title: "Pre-distorted displays" },
  { id: "photorefraction", num: "1.4", title: "Photorefraction sensing" },
  { id: "cnn", num: "1.5", title: "PupilNet estimator" },
  { id: "closed-loop", num: "1.6", title: "Closing the loop" },
  { id: "limits", num: "1.7", title: "Limits of the simulation" },
];

export function Simulation() {
  const [eye, setEye] = useState<Eye>({ S: -2.25, C: -0.75, axis: 30, pupil: 1.5 });
  const [chartDist, setChartDist] = useState(6);
  return (
    <>
      <PageHeader
        kicker="Part 1: simulation"
        title="A digital twin of a refracting eye"
        lede={
          <>
            Before building hardware, we built a simulation of every stage: how a prescription blurs the
            retinal image, what an infrared camera sees in the pupil, how a CNN reads that image back, and
            how a tunable lens converges on the correction. Figures 1 to 7 are computed live in the browser.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="power-vectors" num="1.1" title="Prescriptions as vectors">
          <p>
            A spectacle prescription gives a sphere <Tex>S</Tex>, a cylinder <Tex>C</Tex> and an axis{" "}
            <Tex>\theta</Tex>. That triple is awkward to learn or optimise over: the axis wraps around at
            180°, and the axis is undefined when <Tex>C = 0</Tex>. Following Thibos et al. (1997), every
            stage of eyesim works in <em>power-vector</em> space instead:
          </p>
          <Tex block>{String.raw`M = S + \tfrac{C}{2},\qquad J_0 = -\tfrac{C}{2}\cos 2\theta,\qquad J_{45} = -\tfrac{C}{2}\sin 2\theta`}</Tex>
          <p>
            <Tex>M</Tex> is the spherical equivalent. <Tex>{"(J_0, J_{45})"}</Tex> is a Cartesian astigmatism
            vector. Distances in this space are clinically meaningful, and sums are exact: the residual
            error of an eye wearing a lens is <Tex>{String.raw`\mathbf{p}_{\text{eye}} + \mathbf{p}_{\text{lens}}`}</Tex>.
          </p>
          <Figure num={1} caption="The same prescription in two representations. As the axis passes through 180°, the point in the J-plane returns to its starting position; the power vector has no discontinuity.">
            <PowerVectorFigure eye={eye} onChange={setEye} />
          </Figure>
        </Section>

        <Section id="wave-optics" num="1.2" title="Wavefront to retinal image">
          <p>
            Refractive error is a phase error across the pupil. We map the power vector onto the three
            second-order Zernike modes (defocus and the two astigmatisms), which gives the wavefront
            aberration <Tex>W(x, y)</Tex>. The point-spread function follows from Fraunhofer diffraction:
          </p>
          <Tex block>{String.raw`\mathrm{PSF}(u,v) \;=\; \Bigl|\,\mathcal{F}\bigl\{A(x,y)\,e^{\,i\frac{2\pi}{\lambda}W(x,y)}\bigr\}\Bigr|^2, \qquad I_{\text{retina}} = I_{\text{scene}} \ast \mathrm{PSF}`}</Tex>
          <p>
            Here <Tex>A</Tex> is the circular pupil aperture and <Tex>\lambda = 550</Tex> nm. For large
            blur, the phase wraps faster than a 512² grid can sample. There, the figure switches to the ray-optics
            limit, where each pupil point is deflected by the local wavefront slope. The two
            models agree on blur size and orientation to within a few percent where they overlap.
          </p>
          <p>
            The defocus that reaches the retina is not the prescription itself: a target at distance{" "}
            <Tex>d</Tex> adds a vergence of <Tex>1/d</Tex>, and accommodation (taken as 8 D, a young
            eye) cancels the positive remainder. The figure therefore blurs the chart by{" "}
            <Tex>{String.raw`M + 1/d - A`}</Tex>. A myopic eye sees a distant chart blurred by the full{" "}
            <Tex>|M|</Tex>, but sees the chart sharply once it is brought to the far point at{" "}
            <Tex>{String.raw`1/|M|`}</Tex> metres — a −2.5 D eye reads comfortably at 40 cm. Published
            uncorrected-acuity tables refer to the distant case.
          </p>
          <Figure
            num={2}
            caption={
              <>
                Live optics chain. (a) Wavefront error over the pupil, in waves. (b) Point-spread function,
                log intensity. (c) A letter chart at the chosen distance, convolved with the PSF of the
                residual defocus <Tex>{String.raw`M + 1/d - A`}</Tex>. Letter sizes are Snellen visual
                angles (the bottom row is 20/20), drawn at true angular size for a reader 60 cm from
                this screen. Bring the chart toward a myopic eye's far point and it sharpens; a pure
                cylinder blurs along one meridian only.
              </>
            }
          >
            <OpticsChainFigure eye={eye} onChange={setEye} dist={chartDist} onDist={setChartDist} />
          </Figure>
        </Section>

        <Section id="predistortion" num="1.3" title="Pre-distorted displays">
          <p>
            If the PSF is known, a screen can show an image that the blur will undo. The Wiener filter
            gives the least-squares inverse:
          </p>
          <Tex block>{String.raw`\hat{P} \;=\; \frac{H^{*}}{|H|^{2} + k}\;\hat{I}, \qquad H = \mathcal{F}\{\mathrm{PSF}\}`}</Tex>
          <p>
            The catch is physical. A display can't emit negative light, so the inverse must be clipped to
            [0, 1]. Defocus also puts exact zeros in <Tex>H</Tex>, and those spatial frequencies are lost.
            Pre-distortion helps most for mild error and gives out as blur grows. This is the reason the
            project moves on to correcting the optics themselves.
          </p>
          <Figure num={3} caption="Wiener pre-distortion for the eye and screen distance set in Figure 2. Raise the regulariser k to trade ringing for contrast. Edge energy is the mean squared gradient (the sharpness objective used in §1.6) relative to the unblurred chart.">
            <PredistortFigure eye={eye} dist={chartDist} />
          </Figure>
        </Section>

        <Section id="photorefraction" num="1.4" title="Photorefraction sensing">
          <p>
            To correct an eye automatically, the device first has to measure it. Eccentric
            photorefraction does this with a camera and an infrared LED offset from the lens. Light
            reflects off the retina and leaves the pupil tilted by the eye's own refractive error. Only
            rays that land inside the camera aperture are recorded, which paints a crescent across the
            pupil. For a pupil point <Tex>(x, y)</Tex> and camera distance <Tex>D</Tex>:
          </p>
          <Tex block>{String.raw`\begin{aligned} s_x &= 2\,[(M+J_0)\,x + J_{45}\,y] \\ s_y &= 2\,[(M-J_0)\,y + J_{45}\,x] \end{aligned}\qquad L(x,y) \propto \exp\!\Bigl(-\frac{(s_x D - e_x)^2 + (s_y D - e_y)^2}{2\sigma^2}\Bigr)`}</Tex>
          <p>
            The factor 2 comes from the double pass through the eye. A horizontal LED encodes{" "}
            <Tex>M</Tex> and <Tex>J_0</Tex>, and a vertical LED encodes <Tex>M</Tex> and <Tex>{"J_{45}"}</Tex>, so
            two frames determine the full power vector.
          </p>
          <Figure num={4} caption="Synthetic IR frames for both LEDs (the PupilNet input). The lower row shows the model's fundamental ambiguity: the image depends only on the product M·D.">
            <PhotorefractionFigure eye={eye} />
          </Figure>
          <Note label="Finding.">
            <p>
              Because the frames depend only on <Tex>M\cdot D</Tex>, an unknown camera distance makes{" "}
              <Tex>M</Tex> ambiguous, with an error that grows in proportion to |M|. The camera distance is
              known on the instrument, and the network now receives it as an input together with the LED
              offset, pupil radius and aperture.
            </p>
          </Note>
        </Section>

        <Section id="cnn" num="1.5" title="PupilNet estimator">
          <p>
            PupilNet is a CNN of 152,451 parameters built from depthwise-separable convolutions. It maps
            the two-channel 64×64 photorefraction frame, together with the camera distance, LED offset,
            pupil radius and aperture, to <Tex>{"(M, J_0, J_{45})"}</Tex>. The base network was trained
            for 30 epochs on 300,000 simulated frames with randomised pupil size, distance, eccentricity,
            aberration and sensor noise. The architecture avoids operations that prevent 8-bit
            quantisation, so that the network can run on a microcontroller. Its readings of photographs
            of a physical model eye, with the refraction each photograph represents, are shown on the{" "}
            <Link to="/network#photos">network page</Link> and the <Link to="/model-eye">model-eye page</Link>.
          </p>
          <StatRow>
            <Stat value="0.52" unit="D" label="Test MAE, spherical equivalent M (simulated frames with a visible crescent)" />
            <Stat value="0.09" unit="D" label={<>Test MAE, <Tex>J_0</Tex></>} />
            <Stat value="0.20" unit="D" label={<>Test MAE, <Tex>{"J_{45}"}</Tex></>} />
            <Stat value="300k" label="Simulated training frames" />
          </StatRow>
          <Figure num={5} caption="Training history of the base network (checkpoints/best.pt). The M error is shown twice: over every validation frame, including the blind band, where the image carries no signal and the error cannot fall below the width of the band, and over the frames with a visible crescent, where it reaches 0.52 D.">
            <TrainingFigure />
          </Figure>
          <Figure num={6} caption="Held-out test predictions against ground truth for the base network. Grey points lie in the blind band, where the frame carries no refraction signal.">
            <img
              src="/figures/test_scatter.png"
              alt="Scatter plots of predicted versus true M, J0 and J45 on the synthetic test set"
              style={{ width: "100%", display: "block", background: "#fff" }}
            />
          </Figure>
        </Section>

        <Section id="closed-loop" num="1.6" title="Closing the loop">
          <p>
            The estimate commands a tunable lens. The device then keeps refining, adjusting the lens to
            maximise the sharpness of the retinal image. With no prior, the optimiser first scans
            spherical power from −5 to +5 D to find the right basin, then runs Nelder-Mead over all three
            components. With the CNN, it starts at the negated estimate and only refines.
          </p>
          <Figure num={7} caption="The Step 3 optimiser, run live on the eye from Figure 2 using the retinal sharpness of a Siemens star as its objective. Compare the evaluations each strategy needs to reach the 0.25 D clinical step.">
            <ClosedLoopFigure eye={eye} />
          </Figure>
          <p>
            The CNN estimate in Figure 7 is drawn with the base network's measured test error. Sharpness
            refinement requires a camera that views the scene through the eye's optics, so it applies to
            the bench rather than to the wearable.
          </p>
        </Section>

        <Section id="limits" num="1.7" title="Limits of the simulation">
          <p>
            The figures on this page describe agreement with the simulator, not with real eyes. Comparison
            with photographs of a physical model eye shows that the simulator reproduces the side and growth
            of the crescent and the blind band, while photographed crescents are softer than simulated ones
            and the measured band edges lie 0.3 to 0.8 D further out. The comparison is on the{" "}
            <Link to="/model-eye">model-eye page</Link>. The full-resolution Python pipeline is available in
            the <Link to="/tools/simulate">tools section</Link>.
          </p>
        </Section>
      </Article>
    </>
  );
}
