import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";

const token = process.env.NEUROVISION_WEB_TEST_TOKEN;
const python = process.env.NEUROVISION_WEB_TEST_PYTHON ?? "python";
const api = "http://127.0.0.1:8765";

export default defineConfig({
  testDir: "./tests",
  use: {
    baseURL: "http://127.0.0.1:4173",
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    },
  },
  webServer: [
    ...(token
      ? [
          {
            command: `"${python}" -m uvicorn apps.api.main:create_app --factory --host 127.0.0.1 --port 8765 --no-access-log`,
            cwd: fileURLToPath(new URL("../..", import.meta.url)),
            url: `${api}/api/v1/health`,
            env: { NEUROVISION_API_TOKEN: token },
            reuseExistingServer: false,
          },
        ]
      : []),
    {
      command: "npm run preview -- --port 4173",
      url: "http://127.0.0.1:4173",
      env: token ? { NEUROVISION_WEB_API_TARGET: api } : {},
      reuseExistingServer: false,
    },
  ],
});
