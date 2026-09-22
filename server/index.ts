import { readConfig } from './config.js';
import { createApp } from './app.js';

const config=readConfig();
const {app,store}=createApp(config);
const server=app.listen(config.port,'0.0.0.0',()=>{
  console.log(JSON.stringify({event:'listening',port:config.port,model:config.modelId,trainingStatus:config.trainingStatus,providerConfigured:!!config.inferenceUrl}));
});
server.on('error',(error)=>{console.error(JSON.stringify({event:'listen_failed',error:error.message}));store.close();process.exit(1);});
const prune=setInterval(()=>store.prune(),60000).unref();
let closing=false;
function shutdown(){
  if(closing)return;closing=true;clearInterval(prune);
  server.close(()=>{store.close();process.exit(0);});
  setTimeout(()=>process.exit(1),15000).unref();
}
process.on('SIGTERM',shutdown);process.on('SIGINT',shutdown);
