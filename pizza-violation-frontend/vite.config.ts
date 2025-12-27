// pizza-violation-frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        secure: false,
      },
      "/save": { target: "http://localhost:8000", changeOrigin: true },
      "/start": { target: "http://localhost:8000", changeOrigin: true },
      "/stop": { target: "http://localhost:8000", changeOrigin: true },
      "/violations": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": {
        target: "http://localhost:8000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
