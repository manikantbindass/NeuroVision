import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const target =
  process.env.NEUROVISION_WEB_API_TARGET ?? "http://127.0.0.1:8000";
const proxy = { "/api": { target } };

export default defineConfig({
  plugins: [react()],
  server: { proxy, strictPort: true },
  preview: { proxy, strictPort: true },
});
