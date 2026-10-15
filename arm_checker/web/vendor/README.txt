Three.js 0.180.0 (MIT)
Pinned official npm distribution: https://registry.npmjs.org/three/-/three-0.180.0.tgz
Source project: https://github.com/mrdoob/three.js/tree/r180

Vendored files, unmodified:
  build/three.module.js -> three.module.js
  build/three.core.js -> three.core.js
  examples/jsm/controls/OrbitControls.js -> OrbitControls.js
  LICENSE -> LICENSE

The import map in index.html resolves the bare "three" import locally.
There is no CDN or network dependency at runtime and no Node build step.
