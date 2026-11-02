// Bundles src/ into one self-contained HTML file.
//   "CRIP Physician Dashboard.html"  opens by double-click, no network needed but fonts.
//   dist/artifact.html               the same page in the published-artifact shape.
import { build } from 'esbuild';
import fs from 'node:fs';

const FONTS =
  'https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Mono:wght@400;500;600&family=Atkinson+Hyperlegible+Next:wght@400;500;600;700&display=swap';
const REACT_VERSION = '18.3.1';
const CDN = 'https://cdnjs.cloudflare.com/ajax/libs';

const bundled = await build({
  entryPoints: ['src/app.jsx'],
  bundle: true,
  minify: true,
  write: false,
  format: 'iife',
  target: 'es2020',
  jsx: 'transform',
  jsxFactory: 'React.createElement',
  jsxFragment: 'React.Fragment',
  loader: { '.jpg': 'dataurl' },
  legalComments: 'none',
});

const safe = (js) => js.replaceAll('</script', '<\\/script');
const app = safe(bundled.outputFiles[0].text);
const css = fs.readFileSync('src/styles.css', 'utf8');

const head = `<title>CRIP Physician Dashboard</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="${FONTS}">
<style>${css}</style>`;

const root = '<div id="root"></div>';

fs.mkdirSync('dist', { recursive: true });
fs.writeFileSync(
  'dist/artifact.html',
  `${head}
${root}
<script src="${CDN}/react/${REACT_VERSION}/umd/react.production.min.js"></script>
<script src="${CDN}/react-dom/${REACT_VERSION}/umd/react-dom.production.min.js"></script>
<script>${app}</script>
`,
);

const react = safe(fs.readFileSync('node_modules/react/umd/react.production.min.js', 'utf8'));
const reactDom = safe(fs.readFileSync('node_modules/react-dom/umd/react-dom.production.min.js', 'utf8'));
fs.writeFileSync(
  'CRIP Physician Dashboard.html',
  `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
${head}
</head>
<body>
${root}
<script>${react}</script>
<script>${reactDom}</script>
<script>${app}</script>
</body>
</html>
`,
);

console.log('Built', Math.round(fs.statSync('CRIP Physician Dashboard.html').size / 1024), 'KB');
