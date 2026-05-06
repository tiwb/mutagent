import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildProject } from '@mutgui/core/build-preset';
import project from './mutagent.build.mjs';

const frontendDir = dirname(fileURLToPath(import.meta.url));

await buildProject(frontendDir, project);
