// Exercise the generated viewer script with offline DOM and canvas doubles.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const payload = html.match(/<script type="application\/json" id="payload">([\s\S]*?)<\/script>/);
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/);
assert(payload && script, "Expected the embedded data and viewer script");
const data = JSON.parse(payload[1]);
const elements = new Map();

function element(tag) {
  return {
    tag, value: "", textContent: "", events: {}, children: [], attributes: {},
    append(child) {
      if (child.tag === "option" && this.children.length === 0) this.value = child.value;
      this.children.push(child);
    },
    replaceChildren() { this.children.length = 0; },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, callback) { this.events[name] = callback; },
    getContext() {
      return {
        createImageData: (width, height) => ({
          data: new Uint8ClampedArray(width * height * 4),
        }),
        putImageData: frame => { this.frame = frame.data; },
      };
    },
  };
}
function byId(id) {
  if (!elements.has(id)) elements.set(id, element(id));
  return elements.get(id);
}
byId("payload").textContent = payload[1];
vm.runInNewContext(script[1], {
  document: { getElementById: byId, createElement: element },
  atob: value => Buffer.from(value, "base64").toString("binary"),
}, { timeout: 5000 });

function hasColor(id, rgb) {
  const frame = byId(id).frame;
  for (let i = 0; i < frame.length; i += 4) {
    if (rgb.every((channel, offset) => frame[i + offset] === channel)) return true;
  }
  return false;
}
assert.equal(byId("warning").textContent, data.warning);
assert.equal(byId("notice").textContent, data.notice);
assert.equal(byId("time").children.length, 2);
assert.match(byId("pairSummary").textContent, /Day 0 → 90/);
assert.equal(byId("difference").frame.length, data.shape[0] * data.shape[1] * 4);
assert(hasColor("difference", [68, 224, 218]), "Growing region should show gains");
assert(hasColor("difference", [108, 83, 133]), "Cropped field should be distinct");
const rows = byId("trend").children;
assert.match(rows[0].children[8].textContent, /n=3/);
byId("time").value = "1";
byId("time").events.change();
assert.match(byId("pairSummary").textContent, /Day 0 → 180/);
assert.equal(rows[1].attributes["aria-current"], "true");
assert(hasColor("difference", [255, 170, 80]), "Shrinking region should show losses");

byId("region").value = "300";
byId("region").events.change();
assert.match(byId("trendTitle").textContent, /Stable reference sphere/);
assert(!hasColor("difference", [255, 170, 80]), "Stable region has no losses");
assert(!hasColor("difference", [68, 224, 218]), "Stable region has no gains");
byId("slice").value = "0";
byId("slice").events.input();
assert.equal(byId("sliceIndex").textContent, data.slices[0]);
console.log("NeuroTimeline controls, paired spread, and difference colors passed");
