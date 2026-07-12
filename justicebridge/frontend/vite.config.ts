import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// JusticeBridge frontend — dev server proxies /api to the FastAPI backend
// (justicebridge/api.py, run via `uvicorn justicebridge.api:app --port 8080`)
// so the browser never needs CORS config against a second origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
