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
const TABS = ["overview", "simulator", "ros", "setup", "kria", "fpga", "data", "commands", "status"];
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
        <li><b><code>event_camera</code></b> (<code>ros-events-sim</code>): real GenX320 events, from recordings or the live KV260 stream, into the simulation.</li>
        <li><b>despot</b> on the Kria: Durance's camera client sends the real camera's frames straight to the FPGA; <code>camera_tap</code> reorders them and publishes the picture. No event windows over ROS here.</li>
      </ul>
      <p>Seam in code: <code>reflex/events/io.py</code>, <code>sources.py</code>.</p>`,
  },
  perception: {
    title: "Perception → /reflex/decision",
    body: `<p><b><code>reflex_perception</code></b> takes each window and publishes a <code>ReflexDecision</code>:
      <code>none / left / center / right</code>, a confidence and the time spent. The detector is a parameter:</p>
      <ul>
        <li><code>detector:=snn</code>: the trained spiking network, <code>model:=out/models/snn.pt</code>, <code>device:=cpu|cuda</code>.</li>
        <li><code>detector:=simple</code>: no learning, a burst of events in the lower image is a threat. Calibrates on start
          (<code>calibrate_s</code>); recalibrate with <code>ros2 service call /reflex/recalibrate std_srvs/srv/Trigger</code>.
</li>
        <li><code>detector:=ball</code>: the teammates' ball U-Net on the PC, in the FPGA's integer arithmetic.</li>
      </ul>
      <p><b><code>fpga_perception</code></b> <span class="badge ok">runs</span> on the Kria: reads the spikes of the network on the FPGA
        (zones3 or ball_v3) and publishes the same <code>ReflexDecision</code>, plus the ball's zone on <code>/durance/ball_zone</code>.</p>`,
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
        <li><code>mode:=look</code>: the six zones from the FPGA turn the trunk (column → yaw, row → pitch), feet planted. ball_v3's default on the board.</li>
        <li><code>enabled:=false</code>: logs what it would send and publishes nothing. First thing to run on the real dog.</li>
      </ul>
      <p>The <b><code>follow</code></b> node is the other behavior: walk towards a target, turn the trunk (<code>mode:=gaze</code>) or nine look gestures (<code>ros-look</code>).</p>`,
  },
  robot: {
    title: "Robot ← /api/sport/request",
    body: `<p>Whoever listens on <code>/api/sport/request</code> is the robot.</p>
      <ul>
        <li><b><code>go2_sim_bridge</code></b> with <code>robot_only</code> (<code>ros-robot-sim</code>, <code>ros-drive</code>):
          a MuJoCo Go2 that maps each request to its closest pose after <code>actuation_delay_ms</code> (70 ms, measured on the real Go2). In <code>ros-sim</code>
          it also launches balls and scores each shot on <code>/reflex/sim/outcome</code>.</li>
        <li>Optional walking: <code>locomotion:=rl</code> makes <code>Move</code> requests walk with a learned policy.</li>
        <li>Real Go2: Unitree's sport mode takes the bridge's place. Same message. Every <code>api_id</code> is in
          <code>reflex/runtime/sport_api.py</code>, checked against <code>unitree_ros2</code>; the commands and their signs were tried by hand
          on the real dog on 21 Nov. Our nodes have not driven it yet.</li>
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
    what: "Decision → sport-mode request. Reflex state machine and all safety limits; mode:=look turns the trunk towards the zone.",
    pub: ["/api/sport/request", "/reflex/debug/behavior"], sub: ["/reflex/decision", "/durance/ball_zone"],
    params: "mode, reaction, enabled, min_confidence, consecutive, hold_s, move_vy" },
  { name: "event_camera", exe: "event_camera", kind: "real",
    what: "Real GenX320 events: recordings (.h5 / .bin, looped) or the live KV260 stream, into the simulation.",
    pub: ["/reflex/events_window", "/reflex/events_fullres", "/reflex/debug/events_image"], sub: [],
    params: "source, kria_host, rate_hz, fullres_ms, loop, exclude", note: "Full-resolution frames only while someone subscribes." },
  { name: "camera_tap", exe: "camera_tap", kind: "real",
    what: "On the Kria, between despot and the FPGA server: CHW → HWC, drops frames while the FPGA is busy, camera picture.",
    pub: ["/reflex/debug/events_image"], sub: [], params: "to_hwc, max_inflight, image_hz", note: "Talks TCP to despot (3333) and the server (3334)." },
  { name: "fpga_perception", exe: "fpga_perception", kind: "real",
    what: "On the Kria: the FPGA's spikes → decision, and the ball's zone for six-zone networks.",
    pub: ["/reflex/decision", "/durance/ball_zone", "/reflex/debug/fpga"], sub: [],
    params: "model_dir, readout, margin, vote_window, zone_hold_s, host, port" },
  { name: "target", exe: "target", kind: "real",
    what: "No-learning centroid tracker: events → where the target is (u, v, confidence).",
    pub: ["/reflex/target", "/reflex/debug/target"], sub: ["/reflex/events_window"], params: "–",
    note: "Service /reflex/target/recalibrate." },
  { name: "ball_target", exe: "ball_target", kind: "real",
    what: "The teammates' ball SNN on the CPU (default: the zone classifier) → target.",
    pub: ["/reflex/target", "/reflex/debug/target"], sub: ["/reflex/events_fullres"], params: "checkpoint, min_p, device" },
  { name: "ball_zone_sim", exe: "ball_zone_sim", kind: "tool",
    what: "Stand-in for the board's zone publisher: zones from the keyboard or the CPU SNN.",
    pub: ["/durance/ball_zone"], sub: ["/reflex/events_fullres"], params: "source, checkpoint, vote_window" },
  { name: "follow", exe: "follow", kind: "real",
    what: "Move-stop-look controller: walk bursts, trunk yaw or nine look gestures. Keys f start, SPACE stop, ! Damp, 1-9.",
    pub: ["/api/sport/request", "/reflex/debug/follow"], sub: ["/reflex/target", "/reflex/decision", "/durance/ball_zone"],
    params: "mode, input, dry_run, look_yaw, look_pitch, camera_on_trunk, invert_pitch", note: "Starts disabled. Service /reflex/follow/enable." },
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
    pub: [], sub: ["/reflex/debug/events_image", "/reflex/sim/state", "/reflex/debug/perception", "/reflex/decision", "/durance/ball_zone",
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
  ["ros-follow-sim", "follow_sim.launch.py", { go2_sim_bridge: "y", target: "y" }],
  ["ros-events-sim", "events_sim.launch.py", { go2_sim_bridge: "y", event_camera: "y", target: "y", follow: "y", ball_target: "c", reflex_perception: "c", reflex_behavior: "c" }],
  ["run.sh (Kria)", "kria.launch.py", { camera_tap: "y", fpga_perception: "y", reflex_behavior: "y" }],
  ["ros-follow / ros-look", "ros2 run", { follow: "y" }],
  ["ros-ball-zone-sim", "ros2 run", { ball_zone_sim: "y" }],
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
  return { cam: v("cam"), kinst: v("kinst"), det: v("det"), fmodel: v("fmodel"), robot: v("robot"), scene: v("scene"),
           web: f.web.checked, shell: f.shell.checked };
}

function renderBuilder() {
  const s = builderState();
  const onKria = s.cam === "kria" || s.cam === "event";
  const simRobot = s.robot !== "real";

  // Options that do not apply are hidden or locked.
  const fpga = s.cam === "event";
  $("#fs-kria").hidden = s.cam !== "kria";
  $("#fs-scene").hidden = !simRobot;
  $("#det-fpga").hidden = !fpga;
  $("#det-cpu").hidden = fpga;
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
  if (fpga) {
    const look = s.fmodel === "ball_v3";
    kria.push(["real", "despot + camera_tap", "GenX320 → frames, CHW → HWC"]);
    kria.push(["real", "spleat-server + FPGA", `${s.fmodel} on the FPGA`]);
    kria.push(["real", "fpga_perception", look ? "spikes → zone + decision" : "spikes → decision"]);
    kria.push(["real", "reflex_behavior", look ? "look mode: trunk towards the zone" : "reflex state machine"]);
    if (simRobot) pc.push(["sim", "go2_sim_bridge", `robot only${s.robot === "simweb" ? ", headless" : ", MuJoCo window"}`]);
  } else if (s.cam === "sim") {
    pc.push(["sim", "go2_sim_bridge", "camera + robot, balls, score"]);
    pc.push(["real", "reflex_perception", det]);
    pc.push(["real", "reflex_behavior", "reflex state machine"]);
  } else {
    if (s.cam === "pc") { pc.push(["real", "event_camera_webcam", "webcam → events"]); pc.push(["real", "reflex_perception", det]); }
    if (s.cam === "kria") { kria.push(["real", "event_camera_webcam", "webcam → events"]); kria.push(["real", "reflex_perception", det]); }
    pc.push(["real", "reflex_behavior", s.robot === "real" ? "talks to the real dog" : "reflex state machine"]);
    if (simRobot) pc.push(["sim", "go2_sim_bridge", `robot only${s.robot === "simweb" ? ", headless" : ", MuJoCo window"}`]);
  }
  if (s.robot === "real") go2.push(["plan", "sport mode", "listens on /api/sport/request"]);
  if (s.web && !(fpga && s.robot === "real")) pc.push(["sim", "reflex_web_viewer", "dashboard :8080"]);

  const box = (title, list, badge) => `<div class="machine"><h4>${title}${badge || ""}</h4>${
    list.length ? list.map(([k, n, d]) => `<div class="nd ${k}"><b>${esc(n)}</b><br>${esc(d)}</div>`).join("") : '<div class="empty">not used</div>'}</div>`;
  out.push(`<div class="machines">${box("PC / laptop", pc)}${box("Kria KV260", kria)}${box("Go2", go2, s.robot === "real" ? ' <span class="badge warn">untried</span>' : "")}</div>`);

  const crossing = [];
  if (fpga) {
    if (s.robot === "real") crossing.push("<code>/api/sport/request</code> Kria → Go2 (USB-Ethernet adapter, 192.168.123.x). The laptop sees nothing");
    else crossing.push("<code>/api/sport/request</code> and the camera picture Kria → laptop (cable, 192.168.77.x, <code>dds_interface:=enp8s0</code>)");
  } else {
    if (onKria) crossing.push("<code>/reflex/decision</code> Kria → PC");
    if (s.robot === "real") crossing.push("<code>/api/sport/request</code> PC → Go2 (Ethernet, 192.168.123.x)");
  }
  out.push(`<p class="net-line">${crossing.length ? `Over the network: ${crossing.join(" · ")}. All machines on the same ROS_DOMAIN_ID (default 0).` : "Everything on one machine; nothing crosses the network."}</p>`);

  // ---- warnings and blockers
  if (fpga) out.push(`<p class="net-line">The FPGA must already hold <code>${esc(s.fmodel)}</code>: <code>run.sh</code> refuses a model
    whose output count does not match the server's config. Live decisions are not measured; ${s.fmodel === "ball_v3"
    ? "ball_v3 sees only a ball close to the camera, moved fast" : "zones3 gives brief zone decisions with a ball"}.</p>`);
  if (s.robot === "real") out.push(`<div class="warnbox"><b>Real robot: our nodes have not driven it yet.</b> Emergency stop and the
    Unitree remote in someone's hand, fenced area, rolled balls. Run behavior disabled first (it logs what it would send).
    ${fpga ? "Yaw and pitch of the look gestures are not checked on the real Go2." : "Whether RoboStack's CycloneDDS talks to the Go2 is unverified; the board route above is the one prepared."}</div>`);
  if (s.robot === "simweb" && !s.web) out.push(`<div class="warnbox">Headless simulator and no dashboard: you will only see logs. Tick the browser dashboard.</div>`);
  if (s.robot === "real" && s.web && fpga) out.push(`<p class="net-line">No dashboard with the real dog on this route: the board's DDS leaves on the Go2's adapter only.</p>`);
  if (s.robot === "real" && s.web && !fpga) out.push(`<p class="net-line">With the real dog the dashboard shows events, decisions, reflex state and requests; its 3D dog stays idle (it draws the simulator).</p>`);

  // ---- terminals
  let n = 0;
  const term = (where, title, lines) => { n += 1; out.push(`<div class="terminal-title">Terminal ${n} · ${esc(title)}</div>` + codeBlock(lines, where)); };

  if (fpga) {
    out.push(`<div class="terminal-title">Before: install on the board, check the camera</div>` + codeBlock([
      "# after each code change: cross-compile and copy to /opt/reflex", "scripts/kria/deploy.sh",
      "# after every boot: must end in \"Camera test successfully passed\"", "ssh root@192.168.77.1 despot_test_camera"], "PC"));
  } else if (onKria) {
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
      [`pixi run -e ros ros-sim detector:=${s.det}${park}${viewer}`, "# optional: max_shots:=12 record:=out/loop.mp4 actuation_delay_ms:=70.0"]);
  }
  if (s.cam === "pc") term("PC", "camera side: webcam + perception", [`pixi run -e ros ros-camera detector:=${s.det}`, "# mirror:=true if left and right look swapped; device:=2 for another camera"]);
  if (s.cam === "kria") {
    if (s.kinst === "docker") term("PC → Kria", "camera side on the board (Docker)", [`ssh -t ${KRIA} docker run --rm -it --net=host --device /dev/video0 reflex-kria`]);
    else term("PC → Kria", "camera side on the board (pixi)", [`ssh -t ${KRIA} 'cd hacklab2-durance-go2 && ${PIXI_K} kria-camera detector:=simple'`]);
  }
  if (fpga) {
    if (simRobot) {
      term("PC → Kria", "camera, FPGA, decision and behavior on the board", [`ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=sim model:=${s.fmodel}`]);
      term("PC", "the simulated dog, robot only", [`pixi run -e ros ros-robot-sim behavior:=false dds_interface:=enp8s0${s.robot === "simweb" ? " viewer:=false" : ""}${park}`]);
    } else {
      term("PC → Kria", "dry run: behavior logs what it would send", [`ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=real model:=${s.fmodel}`]);
      term("PC → Kria", "then for real, someone on the remote", [`ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=real model:=${s.fmodel} enabled:=true`]);
    }
  }

  if (s.cam !== "sim" && !fpga) {
    if (simRobot) term("PC", "robot side: behavior + simulated dog", [`pixi run -e ros ros-robot-sim${s.robot === "simweb" ? " viewer:=false" : ""}${park}`, "# options: mode:=dodge reaction:=move locomotion:=rl actuation_delay_ms:=70.0"]);
    else term("PC", "robot side: behavior node only, the Go2 is the robot", [
      "pixi run -e ros ros-shell",
      "# inside the shell: dry run first, it only logs what it would send",
      "ros2 run reflex_ros behavior --ros-args -p enabled:=false",
      "# then for real",
      "ros2 run reflex_ros behavior"]);
  }
  if (s.web && fpga && simRobot) term("PC", "browser dashboard, open http://localhost:8080", [
    "# same DDS settings as the board, on the cable's interface",
    `export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="enp8s0"/></Interfaces></General></Domain></CycloneDDS>'`,
    `pixi run -e ros ros-web${park}`]);
  else if (s.web && !fpga) term("PC", "browser dashboard, open http://localhost:8080", [`pixi run -e ros ros-web${park}`]);
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
  ["Launchers", "PC", "pixi run -e ros ros-sim", "Everything simulated. detector:=snn|simple|ball max_shots:=N actuation_delay_ms:=70.0 record:=out/x.mp4 viewer:=true scene:=park mode:=keeper|dodge"],
  ["Launchers", "PC", "pixi run -e ros ros-camera", "Camera side: webcam + perception. device:=0 detector:=simple|snn mirror:=true"],
  ["Launchers", "PC", "pixi run -e ros ros-robot-sim", "Robot side: behavior + simulated dog. viewer:=false enabled:=false reaction:=move locomotion:=rl"],
  ["Launchers", "PC", "pixi run -e ros ros-drive", "Walking dog driven by /cmd_vel. camera:=true adds the simulated camera + detector; scene:=park"],
  ["Tools", "PC", "pixi run -e ros ros-web", "Browser dashboard at http://localhost:8080. port:=8081 scene:=park. Only listens"],
  ["Tools", "PC", "pixi run -e ros ros-teleop", "Keyboard → /cmd_vel (with ros-drive). i forward, , back, j/l turn, J/L sideways, k stop"],
  ["Tools", "PC", "pixi run -e ros ros-tricks", "Keyboard → built-in tricks. h Hello, s Sit, u RiseSit, ! Damp. allow_acrobatics:=true for flips"],
  ["Tools", "PC", "pixi run -e ros ros-shell", "Shell with ROS 2 ready: ros2 node list, ros2 topic echo, rqt_graph"],
  ["Tools", "PC", "ros2 service call /reflex/recalibrate std_srvs/srv/Trigger", "Recalibrate the no-learning detector (inside ros-shell)"],
  ["Launchers", "PC", "pixi run -e ros ros-events-sim source:=data/river/sat_follow2", "Recorded GenX320 events drive the simulated dog. source:=live kria_host:=root@durance-kria.local for the live camera; behavior:=follow|reflex tracker:=centroid|snn"],
  ["Launchers", "PC", "pixi run -e ros ros-follow-sim", "Simulated dog, its event camera, a scripted ball and the target node. Then ros-follow"],
  ["Launchers", "PC", "pixi run -e ros ros-robot-sim behavior:=false dds_interface:=enp8s0", "The simulated dog as the robot for the Kria's run.sh robot:=sim, over the cable"],
  ["Tools", "PC", "pixi run -e ros ros-follow", "Follow node with a keyboard: f start, SPACE stop, ! Damp. mode:=gaze turns the trunk only. Starts disabled"],
  ["Tools", "PC", "pixi run -e ros ros-look", "Follow node in look mode: nine trunk gestures, never walks; keys 1-9. dry_run:=true"],
  ["Tools", "PC", "pixi run -e ros ros-ball-zone-sim source:=keys", "Stand-in for the board's /durance/ball_zone: 7 8 9 / 1 2 3 a zone, 0 none"],
  ["Real camera", "Kria", "despot_test_camera", "After every boot, before anything else: must end in \"Camera test successfully passed\""],
  ["Real camera", "PC", "python -m capture.record --session NAME --plan cross", "Guided, labelled takes → data/river_samples/NAME. Plans follow, cross, center. Needs river-tools + h5py, not pixi"],
  ["Real camera", "PC", "python -m capture.replay data/river_samples/NAME", "Play takes with label, marks and event rate"],
  ["Real camera", "PC", "python -m capture.relabel data/river_samples/NAME SRC DST --note TEXT", "Move a mislabelled take to the right one"],
  ["Real camera", "PC", "python -m capture.summarize --plan-session NAME", "What is recorded, what is missing"],
  ["Real camera", "PC", "python -m capture.bin_to_h5 FOLDER", "river-tools .bin → .h5"],
  ["FPGA", "PC", "pixi run --manifest-path tools/river/pixi.toml export-ball CKPT.pt models/NAME", "Checkpoint → riverops export + PROVENANCE.md"],
  ["FPGA", "PC", "pixi run spleat-bundle MODEL.json PARAMS.json --firmware models/firmware/JOB", "→ out/spleat/MODEL/: weights.txt, server config, camera config, deploy.yaml. --frequency HZ --no-noise-filter"],
  ["FPGA", "PC", "pixi run --manifest-path tools/river/pixi.toml -e tools river-tools deploy out/spleat/MODEL/deploy.yaml root@192.168.77.1", "Durance's tool: copy firmware and weights to the board"],
  ["FPGA", "PC", "pixi run spleat-read --zones-dir models/zones3", "Read the FPGA's output from the board: winning zone over the last 20 frames"],
  ["Kria", "PC", "scripts/kria/deploy.sh", "Cross-compile reflex_msgs / reflex_ros in Durance's Docker image → /opt/reflex on the board"],
  ["Kria", "PC", "ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=sim model:=ball_v3", "Whole pipeline on the board, simulated dog on the laptop. robot:=real for the Go2 (starts disabled; enabled:=true)"],
  ["Kria", "PC", "scripts/sync_kria.sh ubuntu@kria.local", "Webcam rehearsal board: copy the code and build its Docker image"],
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
