import { Canvas } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { motion, useReducedMotion } from 'motion/react'

function texture(title:string,kind:number){
 const canvas=document.createElement('canvas');canvas.width=540;canvas.height=700
 const ctx=canvas.getContext('2d')!;ctx.fillStyle='#fffefa';ctx.fillRect(0,0,540,700)
 ctx.strokeStyle='#e4e0d7';ctx.lineWidth=1;for(let y=78;y<680;y+=30){ctx.beginPath();ctx.moveTo(36,y);ctx.lineTo(504,y);ctx.stroke()}
 ctx.fillStyle='#5267d8';ctx.font='bold 13px system-ui';ctx.fillText('MODELSLEUTH  /  ILLUSTRATION',38,38)
 ctx.fillStyle='#202a39';ctx.font='600 24px system-ui';ctx.fillText(title,38,76)
 if(kind===0){for(let i=0;i<3;i++)for(let j=0;j<3;j++){ctx.fillStyle=`rgba(82,103,216,${.12+(i===j?.5:.05)})`;ctx.fillRect(95+j*96,160+i*92,86,82);ctx.fillStyle='#2e3b52';ctx.font='16px ui-monospace';ctx.fillText(i===j?'•':'·',132+j*96,205+i*92)}}
 else if(kind===1){ctx.strokeStyle='#5267d8';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(68,480);ctx.lineTo(142,405);ctx.lineTo(228,428);ctx.lineTo(318,282);ctx.lineTo(460,210);ctx.stroke();ctx.fillStyle='#647080';ctx.font='15px system-ui';ctx.fillText('PER-CLASS RECALL',60,170)}
 else if(kind===2){ctx.fillStyle='#e6efe9';ctx.fillRect(38,132,464,132);ctx.fillStyle='#3e8068';ctx.font='bold 15px system-ui';ctx.fillText('OBSERVED EVIDENCE',58,166);ctx.fillStyle='#202a39';ctx.font='18px system-ui';ctx.fillText('A repeated class pattern to inspect',58,214);ctx.fillStyle='#d89a38';ctx.fillRect(38,292,7,146);ctx.fillStyle='#657080';ctx.font='15px system-ui';ctx.fillText('HYPOTHESIS · review boundary examples',58,328)}
 else {ctx.fillStyle='#5267d8';ctx.fillRect(40,150,460,9);ctx.fillStyle='#657080';ctx.font='15px system-ui';for(let i=0;i<6;i++)ctx.fillText(['EXPERIMENT LOG','Saved evaluation','Baseline / revised','Per-class comparison','Group-level review','Validation pending'][i],46,202+i*48)}
 const map=new THREE.CanvasTexture(canvas);map.colorSpace=THREE.SRGBColorSpace;return map
}
function Sheet({index,title,kind,active,onSelect}:{index:number;title:string;kind:number;active:boolean;onSelect:()=>void}){
 const ref=useRef<THREE.Mesh>(null);const map=useMemo(()=>texture(title,kind),[title,kind])
 const positions:[[number,number,number],[number,number,number],[number,number,number],[number,number,number]]=[[-.54,.25,.18],[.45,.26,.04],[-.48,-.28,-.08],[.48,-.25,-.2]]
 return <mesh ref={ref} position={[positions[index][0],positions[index][1],positions[index][2]+(active ? .34 : 0)]} rotation={[0,0,active ? -.045 : positions[index][0]*.12]} onClick={onSelect} onPointerOver={e=>{e.stopPropagation();document.body.style.cursor='pointer'}} onPointerOut={()=>{document.body.style.cursor=''}}><planeGeometry args={[1.86,2.42]}/><meshStandardMaterial map={map} roughness={.96} side={THREE.DoubleSide} color={active?'#fffefa':'#eeeae1'}/></mesh>
}
function Scene({onSelect}:{onSelect:(index:number)=>void}){
 const group=useRef<THREE.Group>(null);const [active,setActive]=useState(2)
 const titles=['Confusion matrix','Class recall','Field note / 04','Experiment log']
 return <group ref={group} position={[0,0,0]} onPointerMove={event=>{if(group.current){group.current.rotation.y=event.point.x*.055;group.current.rotation.x=-event.point.y*.035}}}>{titles.map((title,index)=><Sheet key={title} index={index} title={title} kind={index} active={index===active} onSelect={()=>{setActive(index);onSelect(index)}}/>)}<mesh position={[0,-1.35,-.5]} rotation={[-Math.PI/2,0,0]}><planeGeometry args={[8,5]}/><meshStandardMaterial color="#e8e2d5" roughness={1}/></mesh></group>
}
export default function PaperStack({onSelect}:{onSelect:(index:number)=>void}){
 const reduce=useReducedMotion()
 const [capable,setCapable]=useState(false)
 useEffect(()=>{const query=window.matchMedia('(min-width: 761px) and (prefers-reduced-motion: no-preference)');const update=()=>setCapable(query.matches);update();query.addEventListener('change',update);return()=>query.removeEventListener('change',update)},[])
 return <motion.div className="paper-scene" initial={reduce?false:{opacity:0,y:22}} animate={{opacity:1,y:0}} transition={{duration:reduce?0:.8}}>{capable&&!reduce?<Canvas dpr={[1,1.5]} camera={{position:[0,0,5.8],fov:42}} frameloop="demand" fallback={<div className="scene-fallback">3D preview unavailable. Use the workspace navigation to inspect real analysis.</div>}><ambientLight intensity={1.6}/><directionalLight position={[-3,5,6]} intensity={2.1}/><Suspense fallback={null}><Scene onSelect={onSelect}/></Suspense></Canvas>:<div className="paper-stack-fallback" aria-label="Interactive paper case file previews">{['Confusion matrix','Class recall','Finding evidence','Experiment log'].map((title,i)=><button key={title} onClick={()=>onSelect(i)} className={`preview-sheet preview-sheet-${i}`}><span>MODEL SLEUTH / CONCEPT</span><b>{title}</b><i/></button>)}</div>}<div className="scene-note"><span className="tiny-dot"/> {capable&&!reduce?'Click a sheet · Move pointer to explore':'Select a paper to open its workspace'} <span>·</span> Concept sheets, not live metrics</div></motion.div>
}
