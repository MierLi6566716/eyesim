import { Link } from "react-router-dom";
import {
  Article, Note, PageHeader, Section, Stat, StatRow, articleStyles as a, type TocEntry,
} from "../components/Article";

const TOC: TocEntry[] = [
  { id: "problem", num: "6.1", title: "Accommodation during measurement" },
  { id: "environment", num: "6.2", title: "Simulation environment" },
  { id: "training", num: "6.3", title: "Training" },
  { id: "results", num: "6.4", title: "Results" },
  { id: "scope", num: "6.5", title: "Scope and open questions" },
];

export function Control() {
  return (
    <>
      <PageHeader
        kicker="6 · Control"
        title="Learned lens control for an eye that changes focus"
        lede={
          <>
            During a measurement the eye may change its own focus, and a device that adjusts its lens
            from the measured value can converge on the wrong prescription. A policy trained with
            reinforcement learning to choose lens powers over time reduced the mean error from 0.55 D,
            for the best hand-designed procedure, to 0.46 D on simulated eyes.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="problem" num="6.1" title="Accommodation during measurement">
          <p>
            A young eye adds focusing power involuntarily (accommodation), which makes it measure more
            myopic than it is. In 3,450 children measured without cycloplegic drops the readings were on
            average 0.92 D too myopic (Gu et al., 2022). The clinical countermeasure is fogging, a plus
            lens that blurs the target until accommodation relaxes, but the effective amount and duration
            vary between individuals; in children, 20 minutes of +2 D fogging relaxed accommodation less
            than drops (Hopkins et al., 2012).
          </p>
          <p>
            For a self-adjusting lens the effect is systematic rather than random. If the lens is slightly
            too negative, the eye compensates by accommodating, the image appears focused, and a
            null-seeking controller settles on an over-minus correction. On the 37 cm bench there is an
            additional constraint: mild fogging places the eye inside the photorefraction blind band,
            where no reading is available, so every lens power that produces a reading also disturbs
            the eye.
          </p>
        </Section>

        <Section id="environment" num="6.2" title="Simulation environment">
          <p>One episode is a measurement of up to 10 seconds, simulated at 30 frames per second.</p>
          <ul>
            <li>
              <strong>Eye.</strong> Each simulated person has parameters drawn from published ranges:
              a response latency of 0.25 to 0.45 s, faster accommodation than relaxation, full relaxation
              under mild fogging, drift to a resting focus of 0.25 to 1.75 D under strong fogging,
              instrument myopia that decays only while fogged, depth of focus, and microfluctuations.
            </li>
            <li>
              <strong>Sensor.</strong> Readings are drawn from the network's measured output on the
              model-eye photographs at the corresponding refraction: the measured blind band (&minus;5.8
              to +2.3 D), the measured bias at each refraction, and the measured frame-to-frame noise of
              0.10 D.
            </li>
            <li>
              <strong>Actions.</strong> Change the lens by &minus;3, &minus;1, &minus;0.25, 0, +0.25, +1 or
              +3 D, or end the measurement. The reward is the negative final error, with a small cost per
              frame.
            </li>
            <li>
              <strong>Reported value.</strong> The result is always the median of the last six readings
              plus the lens power. The policy controls where and for how long the eye is observed, not
              the value reported.
            </li>
          </ul>
        </Section>

        <Section id="training" num="6.3" title="Training">
          <p>
            The policy is a two-layer network of 256 units trained with proximal policy optimisation
            (PPO). It observes the last half-second of readings and a short summary of earlier ones.
          </p>
          <ol>
            <li><strong>Training from random initialisation</strong> did not converge (error about 3 D): the reward arrives only at the end of an episode, and random lens changes rarely produce a reading.</li>
            <li><strong>Imitation of a simple rule followed by PPO</strong> degraded performance, because the per-frame cost was large enough to shorten the fogging phase on which the rule depends.</li>
            <li><strong>A reduced time cost, value-function warm-up and a penalty for diverging from the imitated rule</strong> improved on the simple rule (0.84 to about 0.65 D) but not on the stronger one.</li>
            <li><strong>Imitation of the strongest rule</strong> (fog, observe, re-estimate, fog again, observe), followed by PPO, produced the policy reported below.</li>
          </ol>
          <p>
            The best policy was selected on a separate validation set of simulated eyes and evaluated
            once on 2,000 test eyes. The hand-designed rules were written with full knowledge of the
            simulated eye.
          </p>
        </Section>

        <Section id="results" num="6.4" title="Results">
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Procedure (2,000 simulated eyes)</th><th>Mean absolute error</th><th>Errors above 1 D</th><th>No result</th><th>Duration</th></tr></thead>
              <tbody>
                <tr><td>Single reading at the first usable lens</td><td>1.53 D</td><td>&ndash;</td><td>3%</td><td>0.3 s</td></tr>
                <tr><td>Null-seeking loop</td><td>1.41 D</td><td>&ndash;</td><td>0%</td><td>0.4 s</td></tr>
                <tr><td>Fog, then observe</td><td>0.84 D</td><td>31%</td><td>1.9%</td><td>3.5 s</td></tr>
                <tr><td>Fog and observe twice</td><td>0.55 D</td><td>16%</td><td>1.7%</td><td>5.4 s</td></tr>
                <tr><td><strong>Learned policy</strong></td><td><strong>0.46 D</strong></td><td><strong>11%</strong></td><td><strong>0.8%</strong></td><td>5.8 s</td></tr>
                <tr><td>Lower bound: fully relaxed eye, same sensor</td><td>0.34 D</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
              </tbody>
            </table>
          </div>
          <StatRow>
            <Stat value="−0.10" unit="D" label="paired difference from the best rule (95% CI −0.13 to −0.07)" />
            <Stat value="16% to 11%" label="share of errors above 1 D" />
            <Stat value="−0.06" unit="D" label="difference on the 1,961 eyes for which both procedures returned a result" />
          </StatRow>
          <p>
            The improvement comes mainly from fewer failures: the policy halves the rate of measurements
            without a result and reduces large errors, while the median error is unchanged. It closes
            about half of the gap between the best rule and the lower bound.
          </p>
        </Section>

        <Section id="scope" num="6.5" title="Scope and open questions">
          <p>
            Accommodation depends on the individual, age, illumination and refractive state, and no
            simulated population reproduces it exactly. The policy does not predict the eye; it reacts to
            measurements at 30 Hz, so it requires the training population to span the behaviour of real
            eyes rather than to match any one of them. On simulated eyes drawn outside the training
            ranges, all procedures degrade to errors of about 1.8 D, the learned policy among them.
          </p>
          <p>
            Two developments narrow the problem. The <Link to="/prototype">clip-on design</Link> covers the
            measured eye; because accommodation is consensual, its focus then follows the uncovered eye,
            and the lens sweep no longer changes the accommodative stimulus. The remaining control task is
            the fogging of the uncovered eye. Second, the camera now has a video mode intended to record
            the time course of accommodation in a real eye, so that the simulated population can be
            fitted to measurements.
          </p>
          <Note label="Status.">
            <p>
              The improvement is demonstrated in simulation only. Its relevance to human eyes depends on
              how quickly real eyes relax under fogging, which has not yet been measured with this device.
            </p>
          </Note>
        </Section>
      </Article>
    </>
  );
}
