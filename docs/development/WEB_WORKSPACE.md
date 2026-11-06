# NeuroVision web research workspace

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

The web app uses React, TypeScript, and Vite. The guided demo runs entirely in the browser and requires no API token or patient data. It includes two analytic masks, a generated reference, a four-region atlas, and a simulated follow-up on the same voxel grid.

Install Node 22.12+ and use the committed lockfile.

```sh
npm ci --prefix apps/web
npm run dev --prefix apps/web
```

Open [the local workspace](http://127.0.0.1:5173/), then choose **Start guided demo**. The tour covers linked slices, segmentation overlays, a sampled 3D boundary view, confidence, and the timeline. Use the labelled range inputs with the arrow keys for keyboard slice navigation. The image panels show canonical voxel planes. Oblique data is not resliced onto physical cardinal planes. The canvas limits its display sample and the 3D view shows projected boundary voxels.

## Connect a research API

Start the FastAPI process with an operator-provisioned token using the [API setup guide](../api/README.md). The frontend forwards same-origin `/api` requests to `http://127.0.0.1:8000` by default. Set `NEUROVISION_WEB_API_TARGET` in the server environment for a different trusted target. In **System / Models**, connect with the bearer token. Credentials stay in tab memory. Disconnecting clears the case and live results from the UI. Server artifacts follow the backend's storage policy.

Upload reviewed NIfTI volumes and separate reference or atlas label maps. Dataset Explorer shows the first 24 scans returned by the server. Open an MRI to run preprocessing, inference, metrics, NeuroMap, or structured reports. Model Comparison keeps up to six recent predictions for the open case in browser memory. Use the same schema for voxel differences. The backend defaults to a synthetic inference fixture until an operator enables research weights. Confidence is uncalibrated.

The live API supports image and mask exports, QC, single-model inference, atlas association, metrics and JSON reports. The web NeuroTimeline page uses the synthetic case. Use the existing [Python longitudinal workflow](../research/NEUROTIMELINE.md) for independently registered research visits. Observed differences may arise from acquisition, preprocessing, registration, model, or segmentation variation.

## Build and test

```sh
npm run lint --prefix apps/web
npm run format:check --prefix apps/web
npm run build --prefix apps/web
npm exec --prefix apps/web -- playwright install chromium
npm test --prefix apps/web
```

Playwright exercises desktop and mobile navigation, viewer controls, and the guided demo. Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE` to use an installed Chromium-family executable. For the connected browser test, provision a disposable `NEUROVISION_WEB_TEST_TOKEN` (at least 32 characters) and make the API Python environment available. The test starts a temporary API on loopback port 8765. Set `NEUROVISION_WEB_TEST_PYTHON` if its Python executable is not on PATH. Run the build before testing.

For a deployment, serve the Vite `dist` directory and route `/api` to the FastAPI backend on the same origin. The preview server is for local verification. Use the API deployment safeguards before exposing a server to a network.
