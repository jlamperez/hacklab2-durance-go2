"use strict";

/* ------------------------------------------------------------------ helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function copyText(btn, text) {
  const done = () => { btn.textContent = "copied"; btn.classList.add("done");
    setTimeout(() => { btn.textContent = "copy"; btn.classList.remove("done"); }, 1200); };
  if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
}

/* A code block. Lines starting with "#" are drawn as comments and not copied. */
function codeBlock(lines, where, label) {
  const text = lines.filter((l) => !l.startsWith("#")).join("\n");
  const body = lines.map((l) => l.startsWith("#") ? `<span class="comment">${esc(l)}</span>` : esc(l)).join("\n");
  const head = where || label ? `<div class="label">${where ? `<span class="where">${esc(where)}</span>` : ""}${esc(label || "")}</div>` : "";
  return `<div class="code">${head}<pre>${body}</pre><button class="copy" type="button" data-copy="${esc(text)}">copy</button></div>`;
}

/* Copy buttons on static blocks: copy the visible text without comments. */
function addStaticCopyButtons() {
  $$(".code").forEach((block) => {
    if ($(".copy", block)) return;
    const pre = $("pre", block);
    const clone = pre.cloneNode(true);
    $$(".comment", clone).forEach((c) => c.remove());
    const text = clone.textContent.split("\n").map((l) => l.replace(/\s+$/, "")).filter(Boolean).join("\n");
    const b = document.createElement("button");
    b.className = "copy"; b.type = "button"; b.textContent = "copy"; b.dataset.copy = text;
    block.appendChild(b);
  });
}
document.addEventListener("click", (e) => {
  const b = e.target.closest(".copy");
  if (b) copyText(b, b.dataset.copy);
});

/* ------------------------------------------------------------------ theme */
(function theme() {
  const root = document.documentElement;
  try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch (_) {}
  $("#theme-btn").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (_) {}
  });
})();

/* ------------------------------------------------------------------ tabs */
const TABS = ["overview", "simulator", "ros", "setup", "kria", "data", "commands", "status"];
function showTab() {
  const id = location.hash.slice(1);
  const tab = TABS.includes(id) ? id : "overview";
  $$("section.tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${tab}`));
  $$("nav.tabs a").forEach((a) => a.toggleAttribute("aria-current", a.getAttribute("href") === `#${tab}`));
  $$("nav.tabs a").forEach((a) => { if (a.hasAttribute("aria-current")) a.setAttribute("aria-current", "page"); });
  if (TABS.includes(id)) window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", showTab);

/* ------------------------------------------------------------------ pipeline */
const STAGES = {
  source: {
    title: "Event source → /reflex/events_window",
    body: `<p>Publishes one <code>EventWindow</code> per step: the last 50 ms of events binned into 10 time steps,
      2 polarities, 45 × 80 cells. Three programs can play this part:</p>
      <ul>
        <li><b><code>go2_sim_bridge</code></b> (in <code>ros-sim</code>): renders the dog's head camera in MuJoCo and runs the emulator.</li>
        <li><b><code>event_camera_webcam</code></b> (<code>ros-camera</code> on a PC, <code>kria-camera</code> on the board): a webcam's frames through the same emulator. Plumbing test, not a real event camera.</li>
        <li><b>Camera adapter</b> <span class="badge plan">Saturday</span>: <code>camera_webcam_node.py</code> is the template. Replace <code>read_events</code> with the real sensor's SDK; windowing and publishing stay.</li>
      </ul>
      <p>Seam in code: <code>reflex/events/io.py</code>.</p>`,
  },
  perception: {
    title: "Perception → /reflex/decision",
    body: `<p><b><code>reflex_perception</code></b> takes each window and publishes a <code>ReflexDecision</code>:
      <code>none / left / center / right</code>, a confidence and the time spent. The detector is a parameter:</p>
      <ul>
        <li><code>detector:=snn</code>: the trained spiking network, <code>model:=out/models/snn.pt</code>, <code>device:=cpu|cuda</code>.</li>
        <li><code>detector:=simple</code>: no learning, a burst of events in the lower image is a threat. Calibrates on start
          (<code>calibrate_s</code>); recalibrate with <code>ros2 service call /reflex/recalibrate std_srvs/srv/Trigger</code>.
          The only option on the Kria today (no PyTorch there).</li>
        <li>SNN on the FPGA through RIVERLib <span class="badge plan">Saturday</span>: a third backend in <code>reflex/runtime/</code>.</li>
      </ul>`,
  },
  behavior: {
    title: "Behavior → /api/sport/request",
    body: `<p><b><code>reflex_behavior</code></b> runs the reflex state machine and holds all the safety logic.
      It publishes the same <code>unitree_api/Request</code> whether the robot is simulated or real.</p>
      <ul>
        <li>A threat must be seen <code>consecutive:=2</code> times with <code>min_confidence:=0.6</code>.</li>
        <li><code>mode:=keeper</code> blocks the ball, <code>mode:=dodge</code> steps away.</li>
        <li><code>reaction:=euler</code> leans the trunk; <code>reaction:=move</code> steps sideways (needs walking in the simulator: <code>locomotion:=rl</code>).</li>
        <li>Hard limits: lean ≤ 0.5 rad, sideways ≤ 0.5 m/s, watchdog when decisions stop.</li>
        <li><code>enabled:=false</code>: logs what it would send and publishes nothing. First thing to run on the real dog.</li>
      </ul>`,
  },
  robot: {
    title: "Robot ← /api/sport/request",
    body: `<p>Whoever listens on <code>/api/sport/request</code> is the robot.</p>
      <ul>
        <li><b><code>go2_sim_bridge</code></b> with <code>robot_only</code> (<code>ros-robot-sim</code>, <code>ros-drive</code>):
          a MuJoCo Go2 that maps each request to its closest pose after <code>actuation_delay_ms</code> (50 ms, fitted to the real Go2). In <code>ros-sim</code>
          it also launches balls and scores each shot on <code>/reflex/sim/outcome</code>.</li>
        <li>Optional walking: <code>locomotion:=rl</code> makes <code>Move</code> requests walk with a learned policy.</li>
        <li>Real Go2 <span class="badge plan">Saturday</span>: Unitree's sport mode takes the bridge's place. Same message, nothing to change
          in behavior. Every <code>api_id</code> is in <code>reflex/runtime/sport_api.py</code>, checked against <code>unitree_ros2</code>; none tried on a robot yet.</li>
      </ul>`,
  },
};
function showStage(key) {
  $$("#pipeline .stage").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.stage === key)));
  const s = STAGES[key];
  $("#stage-detail").innerHTML = `<b>${esc(s.title)}</b>${s.body}`;
}
$$("#pipeline .stage").forEach((b) => b.addEventListener("click", () => showStage(b.dataset.stage)));

/* ------------------------------------------------------------------ nodes */
const NODES = [
  { name: "go2_sim_bridge", exe: "sim_bridge", kind: "sim",
    what: "MuJoCo Go2, ball launcher, head camera and emulator. Camera and robot at once (ros-sim) or robot only.",
    pub: ["/reflex/events_window", "/reflex/debug/events_image", "/clock", "/reflex/sim/state", "/reflex/sim/outcome", "/reflex/debug/latency_ms"],
    sub: ["/api/sport/request"],
    params: "robot_only, camera, viewer, scene, actuation_delay_ms, max_shots, mode, locomotion, record",
    note: "Camera topics only when it plays the camera; /clock only when it is not robot-only." },
  { name: "event_camera_webcam", exe: "camera_webcam", kind: "real",
    what: "Webcam frames through the emulator: a stand-in event camera. Template for the real camera adapter.",
    pub: ["/reflex/events_window", "/reflex/debug/events_image"], sub: [],
    params: "device, fps, contrast_threshold, mirror" },
  { name: "reflex_perception", exe: "perception", kind: "real",
    what: "Window → decision, with the SNN or the no-learning detector.",
    pub: ["/reflex/decision", "/reflex/debug/perception"], sub: ["/reflex/events_window"],
    params: "detector, model, device, calibrate_s", note: "Service /reflex/recalibrate (simple detector)." },
  { name: "reflex_behavior", exe: "behavior", kind: "real",
    what: "Decision → sport-mode request. Reflex state machine and all safety limits.",
    pub: ["/api/sport/request", "/reflex/debug/behavior"], sub: ["/reflex/decision"],
    params: "mode, reaction, enabled, min_confidence, consecutive, hold_s, move_vy" },
  { name: "cmd_vel_to_sport", exe: "cmd_vel_to_sport", kind: "real",
    what: "Twist on /cmd_vel → Go2 Move request. On the real robot it runs alone.",
    pub: ["/api/sport/request"], sub: ["/cmd_vel"], params: "timeout_s" },
  { name: "teleop_twist_keyboard", exe: "teleop_twist_keyboard", pkg: "teleop_twist_keyboard", kind: "tool",
    what: "Keyboard → /cmd_vel. Started by ros-teleop.", pub: ["/cmd_vel"], sub: [], params: "–" },
  { name: "reflex_tricks", exe: "tricks", kind: "tool",
    what: "Keyboard → built-in tricks (Hello, Sit, dances). Flips only with allow_acrobatics.",
    pub: ["/api/sport/request"], sub: ["/reflex/trick"], params: "allow_acrobatics" },
  { name: "reflex_recorder", exe: "recorder", kind: "tool",
    what: "Records labelled takes from the live window stream into data/real/SESSION.",
    pub: [], sub: ["/reflex/events_window", "/reflex/record/cmd"], params: "session" },
  { name: "reflex_web_viewer", exe: "web_viewer", kind: "tool",
    what: "Browser dashboard (viser) at :8080. Only listens; 3D scene idle without a simulator.",
    pub: [], sub: ["/reflex/debug/events_image", "/reflex/sim/state", "/reflex/debug/perception", "/reflex/decision",
      "/reflex/debug/behavior", "/api/sport/request", "/reflex/debug/latency_ms", "/reflex/sim/outcome"],
    params: "port, scene, seed" },
];
const KIND_BADGE = { sim: '<span class="badge sim">simulator</span>', real: '<span class="badge ok">pipeline</span>', tool: '<span class="badge warn">tool</span>' };

let hlTopic = null;
function renderNodes() {
  const chip = (t) => `<span class="topic-chip${t === hlTopic ? " hl" : ""}" data-topic="${esc(t)}" role="button" tabindex="0">${esc(t)}</span>`;
  $("#node-grid").innerHTML = NODES.map((n) => {
    const touches = n.pub.includes(hlTopic) || n.sub.includes(hlTopic);
    const cls = hlTopic ? (touches ? " hl" : " dim") : "";
    return `<div class="card node-card${cls}">
      <h4>${esc(n.name)} ${KIND_BADGE[n.kind]}</h4>
      <div class="exe">ros2 run ${esc(n.pkg || "reflex_ros")} ${esc(n.exe)}</div>
      <p style="font-size:.88rem;margin:.5rem 0 0">${esc(n.what)}</p>
      <dl><dt>publishes</dt><dd>${n.pub.map(chip).join("") || "–"}</dd>
          <dt>listens</dt><dd>${n.sub.map(chip).join("") || "–"}</dd>
          <dt>params</dt><dd><code style="font-size:.78rem">${esc(n.params)}</code></dd></dl>
      ${n.note ? `<p style="font-size:.8rem;color:var(--muted);margin:.5rem 0 0">${esc(n.note)}</p>` : ""}
    </div>`;
  }).join("");
  if (hlTopic) {
    const pubs = NODES.filter((n) => n.pub.includes(hlTopic)).map((n) => n.name);
    const subs = NODES.filter((n) => n.sub.includes(hlTopic)).map((n) => n.name);
    $("#topic-hl").innerHTML = `<code>${esc(hlTopic)}</code> · published by ${esc(pubs.join(", ") || "nobody here")} · read by ${esc(subs.join(", ") || "nobody here")}`;
  } else $("#topic-hl").textContent = "click a topic";
}
$("#node-grid").addEventListener("click", (e) => {
  const c = e.target.closest(".topic-chip");
  if (c) { hlTopic = hlTopic === c.dataset.topic ? null : c.dataset.topic; renderNodes(); }
});
$("#node-grid").addEventListener("keydown", (e) => {
  const c = e.target.closest(".topic-chip");
  if (c && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); c.click(); }
});
$("#topic-clear").addEventListener("click", () => { hlTopic = null; renderNodes(); });

const LAUNCHERS = [
  ["ros-sim", "sim.launch.py", { go2_sim_bridge: "y", reflex_perception: "y", reflex_behavior: "y" }],
  ["ros-camera / kria-camera", "camera.launch.py", { event_camera_webcam: "y", reflex_perception: "y" }],
  ["ros-robot-sim", "robot_sim.launch.py", { go2_sim_bridge: "y", reflex_behavior: "y" }],
  ["ros-drive", "drive.launch.py", { go2_sim_bridge: "y", cmd_vel_to_sport: "y", reflex_perception: "c" }],
  ["ros-teleop", "ros2 run", { teleop_twist_keyboard: "y" }],
  ["ros-tricks", "ros2 run", { reflex_tricks: "y" }],
  ["ros-record", "ros2 run", { reflex_recorder: "y" }],
  ["ros-web", "ros2 run", { reflex_web_viewer: "y" }],
];
function renderMatrix() {
  const cols = NODES.map((n) => n.name);
  const head = `<tr><th>Command</th><th>Starts</th>${cols.map((c) => `<th style="writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap">${esc(c)}</th>`).join("")}</tr>`;
  const rows = LAUNCHERS.map(([cmd, file, on]) => `<tr><td><code>${esc(cmd)}</code></td><td><code>${esc(file)}</code></td>${
    cols.map((c) => on[c] === "y" ? '<td class="y">●</td>' : on[c] === "c" ? '<td class="c">c</td>' : '<td class="n">·</td>').join("")}</tr>`);
  $("#launch-matrix").innerHTML = head + rows.join("");
}

/* ------------------------------------------------------------------ setup builder */
const KRIA = "ubuntu@kria.local";
const PIXI_K = "~/.pixi/bin/pixi run --frozen -e kria";

function builderState() {
  const f = $("#builder");
  const v = (n) => (f.querySelector(`input[name="${n}"]:checked`) || {}).value;
  return { cam: v("cam"), kinst: v("kinst"), det: v("det"), robot: v("robot"), scene: v("scene"),
           web: f.web.checked, shell: f.shell.checked };
}

function renderBuilder() {
  const s = builderState();
  const onKria = s.cam === "kria" || s.cam === "event";
  const simRobot = s.robot !== "real";

  // Options that do not apply are hidden or locked.
  $("#fs-kria").hidden = !onKria;
  $("#fs-scene").hidden = !simRobot;
  $("#fs-det").hidden = s.cam === "event";
  const snn = $('#builder input[name="det"][value="snn"]');
  snn.disabled = s.cam === "kria";
  snn.closest(".opt").classList.toggle("disabled", snn.disabled);
  if (snn.disabled && snn.checked) { $('#builder input[name="det"][value="simple"]').checked = true; s.det = "simple"; }

  const out = [];
  const park = simRobot && s.scene === "park" ? " scene:=park" : "";

  if (s.cam === "sim" && s.robot === "real") {
    $("#builder-out").innerHTML = `<div class="blocker"><b>Not a valid combination.</b> The simulated camera lives inside
      the simulator: it sees a simulated ball, so it cannot drive a real dog. Pick a webcam or the event camera as the
      source, or a simulated robot.</div>`;
    return;
  }

  // ---- machines and the nodes on them
  const pc = [], kria = [], go2 = [];
  const det = s.det === "snn" ? "SNN" : "no-learning detector";
  if (s.cam === "sim") {
    pc.push(["sim", "go2_sim_bridge", "camera + robot, balls, score"]);
    pc.push(["real", "reflex_perception", det]);
    pc.push(["real", "reflex_behavior", "reflex state machine"]);
  } else {
    if (s.cam === "pc") { pc.push(["real", "event_camera_webcam", "webcam → events"]); pc.push(["real", "reflex_perception", det]); }
    if (s.cam === "kria") { kria.push(["real", "event_camera_webcam", "webcam → events"]); kria.push(["real", "reflex_perception", det]); }
    if (s.cam === "event") { kria.push(["plan", "camera adapter", "real sensor → events (not written)"]); kria.push(["plan", "reflex_perception", "SNN on the FPGA (RIVERLib)"]); }
    pc.push(["real", "reflex_behavior", s.robot === "real" ? "talks to the real dog" : "reflex state machine"]);
    if (simRobot) pc.push(["sim", "go2_sim_bridge", `robot only${s.robot === "simweb" ? ", headless" : ", MuJoCo window"}`]);
  }
  if (s.robot === "real") go2.push(["plan", "sport mode", "listens on /api/sport/request"]);
  if (s.web) pc.push(["sim", "reflex_web_viewer", "dashboard :8080"]);

  const box = (title, list, badge) => `<div class="machine"><h4>${title}${badge || ""}</h4>${
    list.length ? list.map(([k, n, d]) => `<div class="nd ${k}"><b>${esc(n)}</b><br>${esc(d)}</div>`).join("") : '<div class="empty">not used</div>'}</div>`;
  out.push(`<div class="machines">${box("PC / laptop", pc)}${box("Kria KV260", kria)}${box("Go2", go2, s.robot === "real" ? ' <span class="badge warn">untried</span>' : "")}</div>`);

  const crossing = [];
  if (onKria) crossing.push("<code>/reflex/decision</code> Kria → PC");
  if (s.robot === "real") crossing.push("<code>/api/sport/request</code> PC → Go2 (Ethernet, 192.168.123.x)");
  out.push(`<p class="net-line">${crossing.length ? `Over the network: ${crossing.join(" · ")}. All machines on the same ROS_DOMAIN_ID (default 0).` : "Everything on one machine; nothing crosses the network."}</p>`);

  // ---- warnings and blockers
  if (s.cam === "event") out.push(`<div class="blocker"><b>Planned, not runnable yet.</b> Two pieces are missing: the camera
    adapter (<code>camera_webcam_node.py</code> is the template; replace <code>read_events</code> with the sensor's SDK) and
    the RIVERLib backend for perception. Both wait for Durance's hardware and toolchain. Until then, the webcam on the Kria
    rehearses the same layout.</div>`);
  if (s.robot === "real") out.push(`<div class="warnbox"><b>Real robot: not tried yet.</b> Emergency stop first, fenced area,
    rolled balls. Run behavior with <code>enabled:=false</code> before anything else and check the lean direction by hand.
    Whether RoboStack's CycloneDDS talks to the Go2 is unverified.</div>`);
  if (s.robot === "simweb" && !s.web) out.push(`<div class="warnbox">Headless simulator and no dashboard: you will only see logs. Tick the browser dashboard.</div>`);
  if (s.robot === "real" && s.web) out.push(`<p class="net-line">With the real dog the dashboard shows events, decisions, reflex state and requests; its 3D dog stays idle (it draws the simulator).</p>`);

  // ---- terminals
  let n = 0;
  const term = (where, title, lines) => { n += 1; out.push(`<div class="terminal-title">Terminal ${n} · ${esc(title)}</div>` + codeBlock(lines, where)); };

  if (onKria) {
    const lines = s.kinst === "docker"
      ? ["# once, and after every code change: copy the code and build the image on the board", `scripts/sync_kria.sh ${KRIA}`]
      : ["# once, and after every code change: copy the code, then build on the board", `NO_DOCKER=1 scripts/sync_kria.sh ${KRIA}`,
         `ssh -t ${KRIA} 'cd hacklab2-durance-go2 && ${PIXI_K} kria-build'`,
         "# first time only, on the board: install pixi and run `~/.pixi/bin/pixi install -e kria --frozen`"];
    out.push(`<div class="terminal-title">Before: prepare the board</div>` + codeBlock(lines, "PC"));
  }

  if (s.cam === "sim") {
    const viewer = s.robot === "simwin" ? " viewer:=true" : "";
    term("PC", "everything simulated: camera, perception, behavior, robot",
      [`pixi run -e ros ros-sim detector:=${s.det}${park}${viewer}`, "# optional: max_shots:=12 record:=out/loop.mp4 actuation_delay_ms:=50.0"]);
  }
  if (s.cam === "pc") term("PC", "camera side: webcam + perception", [`pixi run -e ros ros-camera detector:=${s.det}`, "# mirror:=true if left and right look swapped; device:=2 for another camera"]);
  if (s.cam === "kria") {
    if (s.kinst === "docker") term("PC → Kria", "camera side on the board (Docker)", [`ssh -t ${KRIA} docker run --rm -it --net=host --device /dev/video0 reflex-kria`]);
    else term("PC → Kria", "camera side on the board (pixi)", [`ssh -t ${KRIA} 'cd hacklab2-durance-go2 && ${PIXI_K} kria-camera detector:=simple'`]);
  }
  if (s.cam === "event") term("Kria", "camera side with the real sensor (planned)", ["# not written yet: a camera adapter node + perception with the RIVERLib backend", "# expected to be started like kria-camera, from a real.launch.py"]);

  if (s.cam !== "sim") {
    if (simRobot) term("PC", "robot side: behavior + simulated dog", [`pixi run -e ros ros-robot-sim${s.robot === "simweb" ? " viewer:=false" : ""}${park}`, "# options: mode:=dodge reaction:=move locomotion:=rl actuation_delay_ms:=50.0"]);
    else term("PC", "robot side: behavior node only, the Go2 is the robot", [
      "pixi run -e ros ros-shell",
      "# inside the shell: dry run first, it only logs what it would send",
      "ros2 run reflex_ros behavior --ros-args -p enabled:=false",
      "# then for real",
      "ros2 run reflex_ros behavior"]);
  }
  if (s.web) term("PC", "browser dashboard, open http://localhost:8080", [`pixi run -e ros ros-web${park}`]);
  if (s.shell) term("PC", "look inside", ["pixi run -e ros ros-shell", "# then: ros2 node list · ros2 topic hz /reflex/decision · rqt_graph"]);

  const expected = [...pc, ...kria].filter(([k]) => k !== "plan").map(([, name]) => name);
  if (expected.length) out.push(`<p class="net-line"><code>ros2 node list</code> should show: ${expected.map((x) => `<code>${esc(x)}</code>`).join(" ")}</p>`);

  $("#builder-out").innerHTML = out.join("");
}
$("#builder").addEventListener("change", renderBuilder);

/* ------------------------------------------------------------------ commands */
const CMDS = [
  ["Setup", "PC", "pixi install -a", "Create the environments, after cloning"],
  ["Setup", "PC", "pixi run setup", "Fetch the Go2 MuJoCo model and Unitree's ROS 2 messages into third_party/"],
  ["Setup", "PC", "pixi run -e ros ros-build", "Build the ROS 2 workspace; again after changing a message, a launch file or setup.py"],
  ["Setup", "PC", "pixi run test", "Unit tests, no ROS needed"],
  ["Simulator", "PC", "pixi run view", "3D window, balls rolled at a standing dog. X stand, C crouch, Q/E lean, Z sit, H hello. --scene park"],
  ["Simulator", "PC", "pixi run drive", "Same window, the dog walks: arrows, A/D sidestep, SPACE stop"],
  ["Simulator", "PC", "pixi run sim", "Write out/demo.mp4: third person, dog's eye, emulated events"],
  ["Simulator", "PC", "pixi run rl-play", "Watch the walking policy in its own simulator (optional extra)"],
  ["Data", "PC", "pixi run gen-data", "2000 simulated episodes → data/synth (~10 min)"],
  ["Data", "PC", "pixi run train", "Train the SNN → out/models/snn.pt"],
  ["Data", "PC", "pixi run eval", "SNN versus the no-learning detector on held-out windows"],
  ["Data", "PC", "pixi run view-data data/synth", "Browse a dataset or play a recording. --label left, --kind crossing, --sheet out/x.png, --video"],
  ["Data", "PC", "pixi run -e ros ros-record session:=sat_am", "Record labelled takes from the live stream. l c r n start, SPACE ball arrived, x cancel, u undo, q quit"],
  ["Data", "PC", "pixi run build-real data/real/*", "Recorded takes → training set data/real_ds"],
  ["Data", "PC", "pixi run finetune-real --val-session sat_pm", "Build + fine-tune on recorded data → out/models/snn_real.pt"],
  ["Launchers", "PC", "pixi run -e ros ros-sim", "Everything simulated. detector:=snn|simple max_shots:=N actuation_delay_ms:=50.0 record:=out/x.mp4 viewer:=true scene:=park mode:=keeper|dodge"],
  ["Launchers", "PC", "pixi run -e ros ros-camera", "Camera side: webcam + perception. device:=0 detector:=simple|snn mirror:=true"],
  ["Launchers", "PC", "pixi run -e ros ros-robot-sim", "Robot side: behavior + simulated dog. viewer:=false enabled:=false reaction:=move locomotion:=rl"],
  ["Launchers", "PC", "pixi run -e ros ros-drive", "Walking dog driven by /cmd_vel. camera:=true adds the simulated camera + detector; scene:=park"],
  ["Tools", "PC", "pixi run -e ros ros-web", "Browser dashboard at http://localhost:8080. port:=8081 scene:=park. Only listens"],
  ["Tools", "PC", "pixi run -e ros ros-teleop", "Keyboard → /cmd_vel (with ros-drive). i forward, , back, j/l turn, J/L sideways, k stop"],
  ["Tools", "PC", "pixi run -e ros ros-tricks", "Keyboard → built-in tricks. h Hello, s Sit, u RiseSit, ! Damp. allow_acrobatics:=true for flips"],
  ["Tools", "PC", "pixi run -e ros ros-shell", "Shell with ROS 2 ready: ros2 node list, ros2 topic echo, rqt_graph"],
  ["Tools", "PC", "ros2 service call /reflex/recalibrate std_srvs/srv/Trigger", "Recalibrate the no-learning detector (inside ros-shell)"],
  ["Kria", "PC", "scripts/sync_kria.sh ubuntu@kria.local", "Copy the code to the board and build its Docker image"],
  ["Kria", "PC", "NO_DOCKER=1 scripts/sync_kria.sh ubuntu@kria.local", "Copy only, for the pixi route"],
  ["Kria", "PC", "ssh -t ubuntu@kria.local docker run --rm -it --net=host --device /dev/video0 reflex-kria", "Run the camera side on the board with Docker"],
  ["Kria", "Kria", "~/.pixi/bin/pixi install -e kria --frozen", "Install the arm64 environment (once, 4.5 min)"],
  ["Kria", "Kria", "~/.pixi/bin/pixi run --frozen -e kria kria-build", "Build the workspace on the board, after each sync"],
  ["Kria", "Kria", "~/.pixi/bin/pixi run --frozen -e kria kria-camera", "Webcam as event camera + detector; same arguments as ros-camera"],
  ["Kria", "Kria", "~/.pixi/bin/pixi run --frozen -e kria kria-shell", "A shell where ros2 works on the board"],
  ["Recipes", "PC", "pixi run -e ros ros-drive camera:=true scene:=park", "Drive the dog in the park and watch its event camera (with ros-teleop and ros-web scene:=park)"],
  ["Recipes", "PC", "ROS_DOMAIN_ID=7 pixi run -e ros ros-sim", "Run a second launcher isolated from the first"],
  ["Recipes", "PC", "pixi run gen-data && pixi run train && pixi run eval", "Retrain from scratch on synthetic data"],
];
const GROUPS = ["All", ...new Set(CMDS.map((c) => c[0]))];
let group = "All";
function renderCommands() {
  const q = $("#cmd-search").value.trim().toLowerCase();
  const hits = CMDS.filter(([g, w, c, d]) => (group === "All" || g === group) && (!q || `${g} ${w} ${c} ${d}`.toLowerCase().includes(q)));
  const by = {};
  hits.forEach((h) => (by[h[0]] = by[h[0]] || []).push(h));
  $("#cmd-list").innerHTML = Object.entries(by).map(([g, rows]) => `<div class="cmd-group"><h3>${esc(g)}</h3><div class="card">${
    rows.map(([, w, c, d]) => `<div class="cmd-row"><div class="cmd-line"><span>${esc(c)}</span><button class="copy" type="button" data-copy="${esc(c)}">copy</button></div>
      <div class="cmd-desc"><span class="badge ${w === "Kria" ? "warn" : "sim"} where">${esc(w)}</span>${esc(d)}</div></div>`).join("")}</div></div>`).join("")
    || '<p class="lede">No command matches.</p>';
}
$("#cmd-pills").innerHTML = GROUPS.map((g) => `<button class="pill" type="button" data-g="${esc(g)}" aria-pressed="${g === group}">${esc(g)}</button>`).join("");
$("#cmd-pills").addEventListener("click", (e) => {
  const p = e.target.closest(".pill"); if (!p) return;
  group = p.dataset.g;
  $$("#cmd-pills .pill").forEach((b) => b.setAttribute("aria-pressed", String(b === p)));
  renderCommands();
});
$("#cmd-search").addEventListener("input", renderCommands);

/* ------------------------------------------------------------------ start */
showStage("source");
renderNodes();
renderMatrix();
renderBuilder();
renderCommands();
addStaticCopyButtons();
showTab();
