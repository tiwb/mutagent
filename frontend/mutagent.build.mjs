import { defineFrontendProject } from '@mutgui/core/build-preset';

export default defineFrontendProject({
  packageName: 'mutagent',
  projectFile: 'mutagent.build.mjs',
  staticDir: '../src/mutagent/static',
  runtimeDirName: 'libs',
  vendorDirName: 'vendor',
  vendors: [],
  runtimes: [
    {
      importName: '@mutagent/ui',
      entry: 'src/index.tsx',
      outFile: 'mutagent-ui.js',
      cssFile: 'mutagent-ui.css',
      peers: ['react', 'react/jsx-runtime', 'antd', '@mutgui/core'],
      kind: 'lib',
    },
  ],
  legacyFiles: [],
});
