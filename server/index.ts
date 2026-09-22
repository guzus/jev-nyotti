import { readConfig } from './config.js';
import { createApp } from './app.js';

const config=readConfig();
const {app,store,scheduler}=createApp(config);
const server=app.listen(config.port,'0.0.0.0',()=>{
  scheduler.start();
  console.log(JSON.stringify({event:'listening',port:config.port,model:config.modelId,trainingStatus:config.trainingStatus,providerConfigured:!!config.inferenceUrl}));
});
server.on('error',(error)=>{console.error(JSON.stringify({event:'listen_failed',error:error.message}));store.close();process.exit(1);});
const prune=setInterval(()=>store.prune(),60000).unref();
let closing=false;
async function shutdown(){
  if(closing)return;closing=true;clearInterval(prune);
  setTimeout(()=>process.exit(1),15000).unref();
  const closed=new Promise<void>(resolve=>server.close(()=>resolve()));
  await Promise.all([closed,scheduler.stop()]);
  store.close();process.exit(0);
}
process.on('SIGTERM',shutdown);process.on('SIGINT',shutdown);
