import { defineConfig } from "@playwright/test";

const desktop = { width: 1440, height: 900 };
const phone = { width: 390, height: 844 };

export default defineConfig({
  testDir: "tests",
  reporter: "list",
  // Reduced motion shows the final state at once, so axe never measures colours mid-fade.
  use: {
    baseURL: "http://localhost:4173",
    browserName: "chromium",
    colorScheme: "dark",
    reducedMotion: "reduce",
  },
  // Tests run against the production build: the same files Vercel serves.
  webServer: {
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: false,
  },
  // `npm test` runs desktop and phone offline (backend blocked, so results are deterministic);
  // `npm run test:live` runs the @live tests against the real backend.
  projects: [
    { name: "desktop", grepInvert: /@live/, use: { viewport: desktop } },
    { name: "phone", grepInvert: /@live/, use: { viewport: phone } },
    { name: "live", grep: /@live/, use: { viewport: desktop } },
  ],
});
