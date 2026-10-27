// A minimal DOM/Canvas event harness. This checks code paths, not browser rendering.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const html = fs.readFileSync(process.argv[2], "utf8");
const embedded = html.match(/<script id="cloud-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(embedded);
const elements = new Map();
let drawn = 0;
for (const id of ["cloud", "zoom", "stats", "angles", "empty", "reset", "cloud-data"]) {
  elements.set(id, { value: "1", textContent: id === "cloud-data" ? embedded : "",
    events: {}, addEventListener(name, callback) { this.events[name] = callback; } });
}
const canvas = elements.get("cloud");
Object.assign(canvas, { clientWidth: 640, clientHeight: 480, setPointerCapture() {}, focus() {},
  getContext() { return { setTransform() {}, fillRect() { drawn++; } }; } });
vm.runInNewContext(script, { document: { getElementById: id => elements.get(id) },
  window: { devicePixelRatio: 1, addEventListener() {} }, requestAnimationFrame: callback => callback() });
assert(drawn > 0);
assert.equal(elements.get("empty").hidden, data.xyz.length > 0);
assert(elements.get("stats").textContent.includes("Units: mm"));
canvas.events.pointerdown({ clientX: 0, clientY: 0, pointerId: 1 });
canvas.events.pointermove({ clientX: 30, clientY: 15 });
assert(!elements.get("angles").textContent.includes("Yaw 0°"));
canvas.events.pointerup();
const before = Number(elements.get("zoom").value);
canvas.events.wheel({ deltaY: -100, preventDefault() {} });
assert(Number(elements.get("zoom").value) > before);
canvas.events.keydown({ key: "ArrowRight", preventDefault() {} });
elements.get("reset").events.click();
assert.equal(Number(elements.get("zoom").value), 1);
assert.equal(elements.get("angles").textContent, "Yaw 0° · Pitch 0°");
console.log(`Canvas harness passed: ${data.xyz.length} points; drag, wheel, keyboard, reset`);
