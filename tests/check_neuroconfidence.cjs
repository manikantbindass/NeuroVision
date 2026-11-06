// Execute the viewer's controls and canvas drawing against an offline DOM harness.
// Run with node tests/check_neuroconfidence.cjs path/to/neuroconfidence.html.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const payload = html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/);
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/);
assert(payload && script, "Expected the embedded data and viewer script");

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    const controls = {
      value: id === "background" ? "hide" : "0",
      textContent: id === "data" ? payload[1] : "",
      style: {},
      events: {},
      children: [],
      frames: [],
      append(child) { this.children.push(child); },
      addEventListener(name, callback) { this.events[name] = callback; },
      getContext() {
        return {
          createImageData: (width, height) => ({
            data: new Uint8ClampedArray(width * height * 4),
          }),
          putImageData: (frame) => this.frames.push(frame),
        };
      },
    };
    elements.set(id, controls);
  }
  return elements.get(id);
}

const context = {
  document: { getElementById: element, createElement: () => element(Symbol()) },
  atob: (value) => Buffer.from(value, "base64").toString("binary"),
  Uint8Array, Uint8ClampedArray, DataView, Math, Array, Number, JSON,
};
vm.runInNewContext(script[1], context, { timeout: 5000 });
const data = JSON.parse(payload[1]);
const [, height, width] = data.shape;
for (const id of ["mri", "segmentation", "neuroconfidence"]) {
  assert.equal(element(id).frames.length, 1);
  assert.equal(element(id).frames[0].data.length, height * width * 4);
}
assert.equal(element("classes").children.length, data.classNames.length);
const initialSummary = element("summary").textContent;
assert.match(initialSummary, /[1-9]\d* displayed voxels/);
assert.match(initialSummary, /[1-9]\d* are low-confidence/);
const initialSlice = element("slice").value;
element("slice").value = "0";
element("slice").events.input();
assert.equal(element("sliceValue").textContent, data.sliceIndices[0]);
element("slice").value = initialSlice;
element("confidence").value = "0";
element("entropy").value = "1";
element("disagreement").value = "1";
element("confidence").events.input();
assert.match(element("summary").textContent, /0 are low-confidence/);
element("reset").events.click();
assert.equal(element("summary").textContent, initialSummary);
assert.equal(Number(element("confidence").value), data.thresholds.confidence);
// Match PyTorch float32 threshold comparison at a rounded boundary such as 0.7.
vm.runInNewContext(
  "confidence.fill(Math.fround(0.7)); entropy.fill(0); disagreement.fill(0);" +
  "byId('confidence').value='0.7'; render();",
  context,
);
assert.match(element("summary").textContent, /0 are low-confidence/);
console.log("Offline viewer controls, threshold reset, legends and canvas frames passed");
