import { mkdirSync,chownSync } from 'node:fs';
import { resolve } from 'node:path';

// Railway mounts volumes as root. Drop privileges before loading app code or secrets.
if(process.getuid?.()===0) {
  const dataDir=resolve(process.env.DATA_DIR??'/data');
  mkdirSync(dataDir,{recursive:true});
  chownSync(dataDir,1000,1000);
  process.setgroups!([]);process.setgid!(1000);process.setuid!(1000);
}
await import('./index.js');
