import { build } from "esbuild";
import { fileURLToPath } from "node:url";

await build({
  absWorkingDir: fileURLToPath(new URL(".", import.meta.url)),
  entryPoints: ["answer-view.js"],
  bundle: true,
  splitting: true,
  format: "esm",
  outdir: "dist",
  minify: true,
  target: "es2022",
  legalComments: "external",
  logLevel: "info",
});
