import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `/api` is proxied to the FastAPI service rather than called cross-origin, so
// the backend needs no CORS middleware. Change the target here, not in the app.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
