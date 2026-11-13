import { Link } from "react-router-dom";
import {
  Article, Note, PageHeader, Section, Stat, StatRow, articleStyles as a, type TocEntry,
} from "../components/Article";

const TOC: TocEntry[] = [
  { id: "problem", num: "6.1", title: "The problem: the eye fights back" },
  { id: "game", num: "6.2", title: "The environment: half real, half simulated" },
  { id: "training", num: "6.3", title: "Training, including the failures" },
  { id: "results", num: "6.4", title: "Results against hand-written rules" },
  { id: "honest", num: "6.5", title: "What this does and does not show" },
];

export function Control() {
  return (
    <>
      <PageHeader
        kicker="6 · Control"
        title="Reinforcement learning against an eye that refocuses"
        lede={
          <>
            The wearer's eye is a second controller acting on the same lens as the device. A learned
            policy that chooses lens powers over time beats the best rule we could write by hand &mdash;
            in simulation, by cutting large misses by a third. This page explains the problem, the agent,
            the four attempts it took, and exactly what remains unproven.
          </>
        }
      />
      <Article toc={TOC}>
        <Section id="problem" num="6.1" title="The problem: the eye fights back">
          <p>
            A young eye can add focusing power at will (accommodation). During a measurement it may focus
            without meaning to, so it reads more myopic than it is: in 3,450 children measured without
            cycloplegic drops, readings averaged 0.92 D too myopic (Gu et&nbsp;al. 2022). Clinics counter
            with <em>fogging</em> &mdash; extra plus power that blurs the view until the eye gives up and
            relaxes &mdash; but the right amount and timing differ per person, and in children even
            20&nbsp;minutes of +2 D fog relaxes less than drops (Hopkins et&nbsp;al. 2012).
          </p>
          <p>
            For a self-setting lens this is worse than noise. If the lens is slightly too minus, the eye
            focuses through it and the image looks fine, so a naive null-seeking loop happily converges on
            an over-minused prescription &mdash; the classic mistake glasses shops avoid by training, and
            devices avoid by protocol. On our bench geometry there is a second twist: the clinically mild
            fog lands the eye inside the photorefraction blind band, where the camera reads nothing at
            all. Every lens that yields a reading also disturbs the eye.
          </p>
        </Section>

        <Section id="game" num="6.2" title="The environment: half real, half simulated">
          <p>The agent plays a 30-frames-per-second game, at most 10 seconds long:</p>
          <ul>
            <li>
              <strong>The patient is simulated</strong>, with every parameter drawn per person from
              published ranges: 0.25&ndash;0.45 s reaction delay, faster focusing than relaxing, full
              relaxation under mild fog, drift to a 0.25&ndash;1.75 D resting focus under heavy fog,
              instrument myopia that fades only while fogged, depth of focus, microfluctuations.
            </li>
            <li>
              <strong>The sensor is real.</strong> Every reading is sampled from what our network actually
              reported on the model-eye photographs at that refraction: the measured blind band
              (&minus;5.8&hellip;+2.3 D), the measured per-setting bias, the measured 0.10 D frame noise.
              No invented camera.
            </li>
            <li>
              <strong>Actions:</strong> change the lens by &minus;3, &minus;1, &minus;0.25, 0, +0.25, +1 or
              +3 D, or stop and report. <strong>Score:</strong> minus the final error, minus a tiny cost
              per frame.
            </li>
            <li>
              <strong>No cheating:</strong> the reported answer is always the median of the last six
              readings plus the lens. The agent cannot learn to guess; it can only choose where to look,
              for how long, and when to stop.
            </li>
          </ul>
        </Section>

        <Section id="training" num="6.3" title="Training, including the failures">
          <p>PPO, a 2&times;256 network, observations = half a second of history plus a small memory. It took four attempts, each one a lesson:</p>
          <ol>
            <li><strong>From scratch:</strong> stuck at ~3 D. Rewards arrive only at the end, and random lens moves almost never find a reading &mdash; nothing to learn from.</li>
            <li><strong>Imitate a simple rule first, then improve:</strong> got <em>worse</em>. Our per-frame time penalty was too big, so the agent shortened the fogging that made the rule work.</li>
            <li><strong>Smaller time cost, value warm-up, a pull toward the imitated rule, honest held-out checks:</strong> beat the simple rule (0.84 &rarr; ~0.65 D) but not a stronger one.</li>
            <li><strong>Imitate the strongest hand rule (fog, flash, re-estimate, fog again), then improve:</strong> the agent below.</li>
          </ol>
          <Note label="Method note.">
            <p>
              The baselines are not strawmen. We wrote the best rules we could &mdash; the two-cycle
              fog-and-flash rule was designed with full knowledge of the simulated eye &mdash; and the agent
              is scored greedily on held-out eyes, with the best checkpoint chosen on a separate validation
              set, never on the test eyes.
            </p>
          </Note>
        </Section>

        <Section id="results" num="6.4" title="Results against hand-written rules">
          <div className={a.tableWrap}>
            <table className={a.table}>
              <thead><tr><th>Policy (2,000 unseen simulated eyes)</th><th>Mean |error|</th><th>Misses &gt; 1 D</th><th>No answer</th><th>Time</th></tr></thead>
              <tbody>
                <tr><td>One shot: find a reading, report</td><td>1.53 D</td><td>&mdash;</td><td>3%</td><td>0.3 s</td></tr>
                <tr><td>Null-seeking loop (the naive device)</td><td>1.41 D</td><td>&mdash;</td><td>0%</td><td>0.4 s</td></tr>
                <tr><td>Fog, then flash (hand rule)</td><td>0.84 D</td><td>31%</td><td>1.9%</td><td>3.5 s</td></tr>
                <tr><td>Fog&ndash;flash twice (best hand rule)</td><td>0.55 D</td><td>16%</td><td>1.7%</td><td>5.4 s</td></tr>
                <tr><td><strong>Learned policy (PPO)</strong></td><td><strong>0.46 D</strong></td><td><strong>11%</strong></td><td><strong>0.8%</strong></td><td>5.8 s</td></tr>
                <tr><td>Floor: eye fully relaxed, same sensor</td><td>0.34 D</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td></tr>
              </tbody>
            </table>
          </div>
          <StatRow>
            <Stat value="−0.10" unit="D" label="agent vs best rule, same eyes (95% CI −0.13 to −0.07)" />
            <Stat value="16% → 11%" label="misses larger than 1 D" />
            <Stat value="−0.06" unit="D" label="advantage on the 1,961 eyes where both answered — part of the gain is simply failing less often" />
          </StatRow>
          <p>
            The agent's edge is reliability, not sharper typical readings: it nearly halves the
            no-answer rate and cuts the worst misses, while the median eye is unchanged. It closes about
            half the gap between the best rule and the physical floor.
          </p>
        </Section>

        <Section id="honest" num="6.5" title="What this does and does not show">
          <p>
            A fair objection: focusing behaviour depends on lighting, age, the person, the prescription
            &mdash; no simulator captures it perfectly, so what is the agent worth? Three answers:
          </p>
          <ul>
            <li>
              <strong>The policy does not need a perfect patient model</strong>, because it never predicts
              the eye open-loop: it reacts to readings, thirty times a second. It needs the <em>family</em>{" "}
              of behaviours in training to cover the person in front of it. That is why every parameter is
              randomized per episode. On eyes drawn <em>outside</em> the training ranges, every policy
              degrades (~1.8 D) &mdash; the agent stays ahead of the rules, but nothing rescues an eye that
              relaxes slower than the session lasts.
            </li>
            <li>
              <strong>The design can remove most of the problem physically.</strong> The{" "}
              <Link to="/prototype">clip-on module</Link> covers the measured eye, so the swept lens no
              longer changes what any eye is trying to focus on; the agent's job then moves to the fellow
              eye's fog. RL is the insurance for the part of the problem the optics cannot delete.
            </li>
            <li>
              <strong>The deciding unknown is measured, not argued.</strong> Everything hinges on how fast
              a real eye relaxes under fog &mdash; seconds or minutes. Our camera's new video mode (about 13
              reading pairs per second end-to-end in a synthetic test) exists to measure exactly that trace
              on a real eye; the simulated patient then gets fitted to data instead of literature ranges.
            </li>
          </ul>
          <Note label="Status.">
            <p>
              The agent's gain is demonstrated in simulation only. We treat it as a hypothesis with a
              measurement plan, not a result about human eyes.
            </p>
          </Note>
        </Section>
      </Article>
    </>
  );
}
