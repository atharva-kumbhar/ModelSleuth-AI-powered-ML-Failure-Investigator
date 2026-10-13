import type { Analysis, DatasetPreview, MappingSuggestion, Sample } from './types'
const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api'
async function request<T>(path:string, init?:RequestInit):Promise<T>{
  let response:Response
  try { response=await fetch(`${BASE}${path}`,init) } catch { throw new Error('Cannot reach the ModelSleuth API. Start the backend and try again.') }
  if(!response.ok){let message=`Request failed (${response.status})`;try{const body=await response.json();message=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)}catch{}throw new Error(message)}
  return response.json() as Promise<T>
}
export const api={
  health:()=>request<{status:string;llm:{configured:boolean;provider:string;model:string;status:string}}>('/health'),
  settings:()=>request<Record<string,any>>('/settings'),
  validate:(file:File)=>{const body=new FormData();body.append('file',file);return request<DatasetPreview>('/datasets/validate',{method:'POST',body})},
  suggestMapping:(dataset_id:string)=>request<MappingSuggestion>('/datasets/suggest-mapping',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id})}),
  analyze:(body:unknown)=>request<Analysis>('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  demo:()=>request<Analysis>('/demo',{method:'POST'}),
  samples:(id:string,params:Record<string,string>)=>request<{total:number;samples:Sample[]}>(`/analysis/${id}/samples?${new URLSearchParams(params)}`),
  explain:(analysis:Analysis,finding:Analysis['findings'][number])=>request<{explanation:{observed_facts:string[];explanation:string;alternative_hypotheses:string[];experiments:string[];validation:string[];limitations:string[];source:string};fallback:boolean;provider_status:string}>('/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({analysis_id:analysis.analysis_id,finding_id:finding.id,summary:{sample_count:analysis.sample_count,metrics:analysis.metrics,class_metrics:analysis.class_metrics,confusion_pairs:analysis.confusion_pairs,groups:analysis.groups,uncertainty:analysis.uncertainty},finding})}),
  chat:async(message:string,history:{role:string;content:string}[],analysisId:string|undefined,files:File[],onToken:(token:string)=>void)=>{const body=new FormData();body.append('message',message);body.append('history',JSON.stringify(history));if(analysisId)body.append('analysis_id',analysisId);files.forEach(file=>body.append('files',file));let response:Response;try{response=await fetch(`${BASE}/chat`,{method:'POST',body})}catch{throw new Error('Cannot reach the ModelSleuth API. Start the backend and try again.')}if(!response.ok){let message=`Request failed (${response.status})`;try{const result=await response.json();message=typeof result.detail==='string'?result.detail:JSON.stringify(result.detail)}catch{}throw new Error(message)}if(!response.body)throw new Error('The chat response could not be streamed.');const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const frames=buffer.split('\n\n');buffer=frames.pop()??'';for(const frame of frames){const event=frame.split('\n').find(line=>line.startsWith('event:'))?.slice(6).trim();const data=frame.split('\n').find(line=>line.startsWith('data:'))?.slice(5).trim();if(!data)continue;if(event==='error'){let message='The AI provider request failed.';try{message=JSON.parse(data).detail??message}catch{}throw new Error(message)}if(data==='[DONE]')return;try{const token=JSON.parse(data).token;if(token)onToken(token)}catch{throw new Error('Could not read the streamed chat response.')}}}if(buffer.trim())throw new Error('The chat response ended unexpectedly.')},
  experiments:()=>request<any[]>('/experiments'),
  saveExperiment:(body:unknown)=>request<any>('/experiments',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  compare:(first_id:number,second_id:number)=>request<any>('/experiments/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({first_id,second_id})}),
}
